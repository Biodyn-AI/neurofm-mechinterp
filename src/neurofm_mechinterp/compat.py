"""Compatibility helpers for importing the upstream BrainOmni code on macOS."""

from __future__ import annotations

import sys
import types


def install_deepspeed_comm_stub() -> None:
    """Install a minimal inference-only deepspeed.comm stub if DeepSpeed is absent."""
    try:
        import deepspeed.comm  # noqa: F401
        return
    except ImportError:
        pass

    if "deepspeed.comm" in sys.modules:
        return

    reduce_op = types.SimpleNamespace(SUM="sum")
    comm = types.SimpleNamespace(
        ReduceOp=reduce_op,
        is_initialized=lambda: False,
        get_world_size=lambda: 1,
        broadcast=lambda tensor, src=0: None,
        all_reduce=lambda tensor, op=None: None,
    )
    module = types.ModuleType("deepspeed")
    module.comm = comm
    sys.modules["deepspeed"] = module
    sys.modules["deepspeed.comm"] = comm

