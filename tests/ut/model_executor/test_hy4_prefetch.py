# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU regressions for the HY4 image's prefetch offload fixes."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.model_executor.offloader.prefetch import (
    _ASCEND_PREFETCH_PARAMETER_ATTR,
    AscendPrefetchOffloader,
    _ModuleOffloader,
    _prefetch_device_loading_context,
)


@pytest.mark.parametrize("has_modules", [False, True])
def test_pageable_graph_check_precedes_upstream_initialization(monkeypatch, has_modules):
    from vllm_ascend.model_executor.offloader import prefetch

    monkeypatch.setattr(prefetch, "should_pin_memory", lambda: False)
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False)),
    )
    offloader = AscendPrefetchOffloader.__new__(AscendPrefetchOffloader)
    offloader.module_offloaders = [object()] if has_modules else []
    with patch("vllm_ascend.model_executor.offloader.prefetch.PrefetchOffloader.post_init") as post_init:
        if has_modules:
            with pytest.raises(RuntimeError, match="Pageable H2D copies cannot be captured"):
                offloader.post_init()
            post_init.assert_not_called()
        else:
            offloader.post_init()
            post_init.assert_called_once_with()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.uint8])
@pytest.mark.parametrize("contiguous", [False, True])
def test_loading_reuses_host_storage_after_transpose(monkeypatch, dtype, contiguous):
    source = (torch.arange(24).reshape(2, 3, 4) % 11).to(dtype)
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(source.clone(), requires_grad=False)
    setattr(module.weight, _ASCEND_PREFETCH_PARAMETER_ATTR, True)
    pointer = module.weight.data_ptr()
    original_to = torch.Tensor.to

    def emulate_device_copy(tensor, *args, **kwargs):
        if args and args[0] == torch.device("npu:0"):
            return tensor.clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", emulate_device_copy)
    fallback = Mock(side_effect=AssertionError("Marked weights must reuse host storage"))
    with _prefetch_device_loading_context(module, torch.device("npu:0"), fallback=fallback):
        module.weight.data = module.weight.data.transpose(1, 2)
        if contiguous:
            module.weight.data = module.weight.data.contiguous()

    assert module.weight.data_ptr() == pointer
    torch.testing.assert_close(module.weight.float(), source.transpose(1, 2).float(), rtol=0, atol=0)
    fallback.assert_not_called()


def test_sync_prunes_empty_absorbed_projections_before_upstream_sync():
    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    live = SimpleNamespace(_param=torch.nn.Parameter(torch.ones(2)))
    empty = SimpleNamespace(_param=torch.nn.Parameter(torch.empty(0)))
    offloader._param_offloaders = {"live": live, "absorbed": empty}

    with patch("vllm_ascend.model_executor.offloader.prefetch.VllmModuleOffloader.sync_cpu_storage") as sync:
        offloader.sync_cpu_storage()
        sync.assert_called_once_with()
    assert offloader._param_offloaders == {"live": live}


def test_deleted_quantizer_parameters_are_left_for_upstream_pruning():
    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    deleted = SimpleNamespace()
    offloader._param_offloaders = {"deleted_scale": deleted}

    with patch("vllm_ascend.model_executor.offloader.prefetch.VllmModuleOffloader.sync_cpu_storage") as sync:
        offloader.sync_cpu_storage()
        sync.assert_called_once_with()
    offloader._capture_static_buffer_formats_from_npu_params()
    assert offloader._param_offloaders == {"deleted_scale": deleted}


@pytest.mark.parametrize("transpose", [False, True])
def test_static_copy_preserves_dense_weight_layout(transpose):
    source = torch.arange(24).reshape(2, 3, 4)
    if transpose:
        source = source.transpose(1, 2)
    destination = torch.empty_strided(source.shape, source.stride(), dtype=source.dtype)
    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    offloader._copy_param_to_static(source, destination)
    torch.testing.assert_close(destination, source)


def test_prefetch_joins_copy_stream_and_records_completion(monkeypatch):
    from vllm_ascend.model_executor.offloader import prefetch

    events = Mock()
    fork = events.fork
    compute = events.compute
    copy = events.copy
    monkeypatch.setattr(prefetch.torch.npu, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(prefetch.torch.npu, "Event", lambda: fork)
    monkeypatch.setattr(prefetch.torch.npu, "current_stream", lambda: compute)
    monkeypatch.setattr(prefetch.torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(prefetch, "should_pin_memory", lambda: False)
    offloader = _ModuleOffloader.__new__(_ModuleOffloader)
    offloader._buffer_pool = object()
    offloader.copy_stream = copy
    offloader._copy_done_event = events.done
    offloader._param_offloaders = {"weight": SimpleNamespace(_cpu_storage=torch.ones(2), _gpu_buffer=torch.zeros(2))}
    offloader._copy_param_to_static = events.transfer

    offloader.start_onload_to_static()

    names = [str(call).split("(", 1)[0] for call in events.mock_calls]
    assert names == ["call.compute.record_event", "call.copy.wait_event", "call.transfer", "call.done.record"]
    assert offloader._prefetch_in_capture
    assert not offloader._event_valid_for_eager
