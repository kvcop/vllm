# Adaptive Verification

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. At batch size 256 it is a much more delicate one. Draft tokens now compete with real tokens for the same compute, and every rejected token is compute wasted; with enough of them, throughput drops.

That matters because per-position acceptance decays fast. While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost. The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies.

Adaptive verification decides per step how much of the draft to verify instead. Every (request, position) draft slot is scored by its *survival probability*, the running product of that request's per-position confidences, and the highest-scoring slots are admitted until a global budget is spent. Slots compete across requests: position 5 of a confident request can outrank position 1 of a doubtful one, so one request keeps its full block while another could be trimmed after a token or two.

The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.

The practical effect is that one configuration holds up across the whole load range, which removes most of the need to tune `num_speculative_tokens` per deployment.

## Support

Adaptive verification needs per-position acceptance estimates, so today its currently only supported for DSpark with a **confidence head**.

## Usage

It is enabled by default for DSpark:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": true
  }'
```

Set `enable_adaptive_verification: false` to verify the full block for every request.

For DSV4-style checkpoints, `num_speculative_tokens` must be at least the checkpoint's `dspark_block_size` (for example, 7 for DeepSeek-V4-Flash). For Qwen3 DSpark checkpoints in speculators format, it must equal `block_size` when `sample_from_anchor=true`; legacy checkpoints with `sample_from_anchor=false` use the anchor as a bonus token and require `block_size - 1`. Their Markov head was trained for that fixed semi-autoregressive layout. Both contracts are validated at startup.

### Confidence calibration

`dspark_confidence_temperatures` optionally supplies one positive, finite temperature per speculative position. vLLM applies it only to the confidence logit before the sigmoid, `sigmoid(logit / temperature[position])`; it never changes proposal sampling or rejection-sampling temperatures. Omit the option (the default) to use the checkpoint's raw confidence scores. An identity vector is useful only as a raw-confidence diagnostic:

```json
"dspark_confidence_temperatures": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

Use calibrated values only after fitting them from held-out left-to-right sequential temperature scaling (STS) data with the same target, draft, and sampling configuration.

The production scheduler uses confidence scores from exactly two decode steps earlier (`t−2`) to select the next global verification budget. This preserves a fixed captured graph for the current target forward; it is distinct from the paper's synchronous all-candidate allocation algorithm.

### Collecting calibration data

DSpark can collect bounded, payload-free calibration shards while adaptive verification is disabled. Collection is intentionally strict: it requires full-block verification, probabilistic drafting and rejection sampling, and either omitted or all-ones confidence temperatures. For example:

```json
{
  "method": "dspark",
  "model": "RedHatAI/Qwen3.6-35B-A3B-speculator.dspark",
  "num_speculative_tokens": 8,
  "draft_sample_method": "probabilistic",
  "enable_adaptive_verification": false,
  "dspark_confidence_capture_path": "/private/dspark-calibration",
  "dspark_confidence_capture_max_rows": 100000,
  "dspark_confidence_capture_shard_rows": 4096,
  "dspark_confidence_temperatures": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
```

Each data-parallel rank writes to its own private `dp-NNNN` directory. Only TP rank zero on the last pipeline stage is elected as its writer. Shards have mode `0600`, stop at the configured hard row cap, and contain only:

- `raw_logits`: `float32[N,K]`, captured before temperature scaling and sigmoid;
- `prefix_mask`: `uint8[N,K]` prefix-survival labels;
- `verified_lengths`, `accepted_counts`: `uint8[N]`;
- `request_ordinal`: opaque `uint64[N]` values;
- `proposal_seq`: `uint32[N]` and `engine_step`: `uint64[N]`.

For each row, only positions below `verified_lengths` are observed; later positions are censored and must not be treated as negative labels. The capture never stores request IDs, token IDs, prompts, responses, hidden states, or target/draft vocabulary logits.

Capture performs confidence-head compute, device-to-host copies, and synchronous shard writes. Exclude capture runs from throughput, TTFT, latency, and other performance conclusions.

## Requirements and limitations

- The attention backend must tolerate device-decided query lengths, since the CPU lengths only bound them from above. Backends that plan off the CPU lengths are excluded by the attention selector, and rejected at startup for models that hard-wire their backend.
- Full cudagraphs are required: step costs are profiled from captured graphs, so `--enforce-eager` is rejected at startup.
- Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries), pipeline parallelism (cost curves and confidences exist only on the last rank), or output logprobs (to be fixed).

## Tuning the cost profile

Step costs are profiled against a synthetic KV context, 8192 tokens by default. Deployments serving much longer contexts may want to raise it so the profiled step reads a more realistic amount of cache (this matters a bit less for sparse attention models like DeepSeek-v4 since the cheap indexer is the main cost that scales with context length).

```bash
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=131072
```
