# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

from vllm.model_executor.model_loader import utils


class MarlinFP8ScaledMMLinearKernel:
    pass


class ModelOptFp8LinearMethod:
    def __init__(self) -> None:
        self.fp8_linear = MarlinFP8ScaledMMLinearKernel()


def test_route_audit_reads_modelopt_fp8_kernel(monkeypatch) -> None:
    records: list[dict[str, object]] = []
    layer = SimpleNamespace(quant_method=ModelOptFp8LinearMethod())
    model = SimpleNamespace(named_modules=lambda: [("layer", layer)])

    monkeypatch.setenv("VLLM_NVFP4_ROUTE_AUDIT", "1")
    monkeypatch.setattr(
        utils.logger,
        "info",
        lambda _message, payload: records.append(json.loads(payload)),
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_rank", lambda: 1
    )

    utils._log_nvfp4_route_audit(model)

    assert records == [
        {
            "fallback": None,
            "full_weight_shape": None,
            "input_quantization": "bf16-a16",
            "kernel": "MarlinFP8ScaledMMLinearKernel",
            "method": "ModelOptFp8LinearMethod",
            "module": "layer",
            "partition_weight_shape": None,
            "tiled": None,
            "tp_rank": 1,
        }
    ]
