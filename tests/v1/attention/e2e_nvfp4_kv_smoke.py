# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end generation smoke for the NVFP4 KV cache (TRITON_ATTN backend).

Runs one engine per --kv-cache-dtype on 20 fixed prompts (greedy, top-1
logprobs) and dumps a JSON artifact; ``--compare`` diffs the artifacts.

  python tests/v1/attention/e2e_nvfp4_kv_smoke.py --dtype nvfp4 \
      --out .tmp/e2e/nvfp4.json
  python tests/v1/attention/e2e_nvfp4_kv_smoke.py --compare \
      .tmp/e2e/auto.json .tmp/e2e/fp8.json .tmp/e2e/nvfp4.json

Defaults to Qwen/Qwen3-0.6B (downloads on first use); any small dense
checkpoint fits.  enforce_eager keeps the run off torch.compile so the
comparison isolates the KV cache dtype.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The first three prime numbers are",
    "A rectangle has how many sides?",
    "In Russian, 'thank you' is",
    "The chemical formula of table salt is",
    "The largest planet in the solar system is",
    "Python is a programming language that",
    "Photosynthesis converts sunlight into",
    "The speed of light in vacuum is about",
    "A triangle's angles always sum to",
    "The freezing point of water in Fahrenheit is",
    "Mount Everest is located in the",
    "Binary numbers use only the digits",
    "The human heart has how many chambers?",
    "Gravity pulls objects toward",
    "Shakespeare wrote the play called",
    "The ocean covers most of the Earth's",
    "Honey is produced by",
    "The Roman numeral for fifty is",
]


def run_dtype(dtype: str, out_path: str, model: str) -> None:
    from vllm import LLM, SamplingParams
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    llm = LLM(
        model=model,
        kv_cache_dtype=dtype,
        attention_backend=AttentionBackendEnum.TRITON_ATTN,
        dtype="float16",
        gpu_memory_utilization=0.45,
        max_model_len=768,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=48, logprobs=1, seed=0)
    outs = llm.generate(PROMPTS, params)
    records = []
    for req_out in outs:
        top1 = []
        for tok_lp in req_out.outputs[0].logprobs:
            lp = next(iter(tok_lp.values()))
            top1.append(lp.logprob)
        records.append(
            {
                "prompt": req_out.prompt,
                "token_ids": list(req_out.outputs[0].token_ids),
                "text": req_out.outputs[0].text,
                "top1_logprobs": top1,
            }
        )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"dtype": dtype, "records": records}))
    print(f"WROTE {out_path} ({len(records)} prompts)")


def compare(paths: list[str]) -> None:
    data = [json.loads(Path(p).read_text()) for p in paths]
    base = data[0]
    for other in data[1:]:
        exact = 0
        deltas = []
        min_len_mismatch = 0
        for b, o in zip(base["records"], other["records"]):
            if b["token_ids"] == o["token_ids"]:
                exact += 1
            else:
                min_len_mismatch += 1
            for lb, lo in zip(b["top1_logprobs"], o["top1_logprobs"]):
                deltas.append(abs(lb - lo))
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        print(
            f"{other['dtype']}-vs-{base['dtype']}: exact_match={exact}/20 "
            f"mismatched_seqs={min_len_mismatch} "
            f"mean_abs_dlogprob={mean_delta:.6f} "
            f"max_abs_dlogprob={max(deltas) if deltas else 0:.6f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["auto", "fp8", "nvfp4"])
    ap.add_argument("--out")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        assert args.dtype and args.out, "--dtype and --out required"
        run_dtype(args.dtype, args.out, args.model)
