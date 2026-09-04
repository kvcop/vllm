# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-only tests: an explicit offloading connector rejects nvfp4 KV.

``_post_init_kv_transfer_config`` returns early when kv_offloading_size is
None, so an explicit ``--kv-transfer-config`` with an offloading connector
(OffloadingConnector / SimpleCPUOffloadConnector / LMCache*, including
MultiConnector children) previously bypassed the nvfp4 rejection until the
engine crashed later. ``_verify_kv_transfer_config`` runs unconditionally
for any configured connector, which is where the guard lives now.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig, VllmConfig


def _stub(
    cache_dtype: str, connector: str, children: list[dict] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype, kv_offloading_size=None
        ),
        model_config=None,
        kv_transfer_config=KVTransferConfig(
            kv_connector=connector,
            kv_role="kv_both",
            kv_connector_extra_config={"connectors": children} if children else {},
        ),
    )


def test_explicit_offloading_connector_rejected_with_nvfp4() -> None:
    with pytest.raises(ValueError, match="not supported with an NVFP4 KV cache"):
        VllmConfig._verify_kv_transfer_compat(_stub("nvfp4", "OffloadingConnector"))


def test_explicit_simple_cpu_offload_rejected_with_nvfp4() -> None:
    with pytest.raises(ValueError, match="SimpleCPUOffloadConnector"):
        VllmConfig._verify_kv_transfer_compat(
            _stub("nvfp4", "SimpleCPUOffloadConnector")
        )


def test_lmcache_connector_rejected_with_nvfp4() -> None:
    with pytest.raises(ValueError, match="LMCacheMPConnector"):
        VllmConfig._verify_kv_transfer_compat(_stub("nvfp4", "LMCacheMPConnector"))


def test_multi_connector_offloading_child_rejected_with_nvfp4() -> None:
    children = [
        {"kv_connector": "SimpleCPUOffloadConnector"},
        {"kv_connector": "NixlConnector"},
    ]
    with pytest.raises(ValueError, match="SimpleCPUOffloadConnector"):
        VllmConfig._verify_kv_transfer_compat(
            _stub("nvfp4", "MultiConnector", children)
        )


def test_non_offloading_connector_and_non_nvfp4_dtype_pass() -> None:
    VllmConfig._verify_kv_transfer_compat(_stub("fp8", "OffloadingConnector"))
    VllmConfig._verify_kv_transfer_compat(_stub("nvfp4", "NixlConnector"))
    VllmConfig._verify_kv_transfer_compat(
        _stub("nvfp4", "MultiConnector", [{"kv_connector": "NixlConnector"}])
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
