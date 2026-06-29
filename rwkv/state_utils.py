from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def normalize_model_arg(model_arg: str) -> tuple[str, str]:
    path = Path(model_arg)
    if model_arg.endswith(".pth"):
        return model_arg[:-4], model_arg
    if path.is_file():
        return model_arg[:-4] if model_arg.endswith(".pth") else model_arg, model_arg
    return model_arg, model_arg + ".pth"


def torch_load_weights(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def freeze_except_time_state(model: torch.nn.Module) -> int:
    count = 0
    for name, param in model.named_parameters():
        if "time_state" in name or "ts_state" in name:
            param.data = param.data.float()
            param.requires_grad = True
            count += param.numel()
        else:
            param.requires_grad = False
    return count


def save_time_state_only(model: torch.nn.Module, path: str) -> None:
    state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "time_state" in name or "ts_state" in name
    }
    torch.save(state, path)


def load_time_state_only(model: torch.nn.Module, path: str) -> bool:
    path_obj = Path(path)
    if not path or not path_obj.exists():
        return False
    state = torch_load_weights(str(path_obj))
    if "full_state" in state and isinstance(state["full_state"], dict):
        state = state["full_state"]
    elif "time_state" in state and isinstance(state["time_state"], dict):
        state = state["time_state"]

    hits = 0
    for name, param in model.named_parameters():
        value = state.get(name)
        if torch.is_tensor(value):
            param.data.copy_(value.to(device=param.device, dtype=param.dtype))
            hits += 1
    return hits > 0


def init_runtime_state_with_time_state(infer_model, train_model, batch_size: int, device: str):
    state = infer_model.generate_zero_state(batch_size)
    for layer_idx, block in enumerate(train_model.blocks):
        ts = block.att.time_state
        state[1][layer_idx] = (
            ts.to(device=state[1].device, dtype=state[1].dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .clone()
        )
        if hasattr(block.att, "ts_state"):
            state[0][layer_idx, 0] = (
                block.att.ts_state.to(device=state[0].device, dtype=state[0].dtype)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .clone()
            )
        if hasattr(block.ffn, "ts_state"):
            state[0][layer_idx, 1] = (
                block.ffn.ts_state.to(device=state[0].device, dtype=state[0].dtype)
                .unsqueeze(0)
                .expand(batch_size, -1)
                .clone()
            )
    return state
