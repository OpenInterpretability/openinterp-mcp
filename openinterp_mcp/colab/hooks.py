"""Forward-hook utilities for capturing residual-stream activations.

Hooks register on `model.model.layers[i]` (LlamaDecoderLayer-style; works for Qwen, Llama,
Gemma decoder layers). The layer output's first element is the residual stream after the layer.
For atypical architectures, callers can pass a custom `layer_module_resolver`.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List, Optional


def default_layer_module(model, idx: int):
    """Return the module whose `output[0]` is the post-block residual stream."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[idx]
    raise RuntimeError(
        "Could not locate decoder layers on model. Pass a custom layer_module_resolver to capture()."
    )


@contextmanager
def capture_layers(
    model,
    layer_indices: List[int],
    layer_module_resolver: Optional[Callable] = None,
) -> Iterator[Dict[int, "torch.Tensor"]]:
    """Context manager that captures the residual-stream tensor at each requested layer.

    Usage:
        with capture_layers(model, [10, 20]) as buf:
            model(input_ids)
        buf[10].shape == (batch, seq, d_model)
    """
    resolver = layer_module_resolver or default_layer_module
    buf: Dict[int, "torch.Tensor"] = {}
    handles = []

    def make_hook(i: int):
        def _hook(_mod, _inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            buf[i] = tensor.detach()
        return _hook

    try:
        for i in layer_indices:
            module = resolver(model, i)
            handles.append(module.register_forward_hook(make_hook(i)))
        yield buf
    finally:
        for h in handles:
            h.remove()


@contextmanager
def steer_layer(
    model,
    layer_idx: int,
    direction: "torch.Tensor",
    alpha: float,
    apply_at_positions: Optional[List[int]] = None,
    layer_module_resolver: Optional[Callable] = None,
):
    """Forward-hook that adds `direction * alpha` to the residual stream at `layer_idx`.

    If `apply_at_positions` is None, steering is applied to every token position.
    Otherwise only the listed positions are modified — useful for surgical interventions.
    """
    import torch

    resolver = layer_module_resolver or default_layer_module
    module = resolver(model, layer_idx)
    direction = direction.to(dtype=torch.float32)

    def _hook(_mod, _inp, out):
        is_tuple = isinstance(out, tuple)
        tensor = out[0] if is_tuple else out
        delta = (direction * alpha).to(dtype=tensor.dtype, device=tensor.device)
        if apply_at_positions is None:
            modified = tensor + delta
        else:
            modified = tensor.clone()
            for pos in apply_at_positions:
                modified[..., pos, :] = modified[..., pos, :] + delta
        if is_tuple:
            return (modified,) + out[1:]
        return modified

    handle = module.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()
