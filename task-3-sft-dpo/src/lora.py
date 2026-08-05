"""不依赖 peft 的 LoRA 注入、保存与合并实现。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """冻结原始 ``nn.Linear``，并叠加 ``(alpha / r) * B(Ax)`` 的低秩更新。"""

    def __init__(self, base_layer: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r 必须为正数")
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r, **factory_kwargs))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + update * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """将低秩矩阵写回一个普通 Linear，便于无 LoRA 分支推理。"""
        base = self.base_layer
        merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                           device=base.weight.device, dtype=base.weight.dtype)
        merged.weight.copy_(base.weight + (self.lora_B @ self.lora_A).to(base.weight.dtype) * self.scaling)
        if base.bias is not None:
            merged.bias.copy_(base.bias)
        return merged


def _replace_child(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = child
    else:
        setattr(parent, child_name, child)


def inject_lora(model: nn.Module, target_modules: Iterable[str], r: int = 8,
                alpha: float = 16.0) -> nn.Module:
    """冻结模型后，按末级模块名把目标 Linear 替换为 :class:`LoRALinear`。"""
    targets = set(target_modules)
    for parameter in model.parameters():
        parameter.requires_grad = False

    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for full_name, module in model.named_modules():
        if not full_name or not isinstance(module, nn.Linear):
            continue
        if full_name.rsplit(".", 1)[-1] not in targets:
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child_name, module))

    if not replacements:
        raise ValueError(f"未找到 LoRA 目标 Linear 层：{sorted(targets)}")
    for parent, child_name, module in replacements:
        _replace_child(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """把全部 LoRA 更新合并入基础 Linear，并移除 LoRA 分支。"""
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []
    for full_name, module in model.named_modules():
        if not full_name or not isinstance(module, LoRALinear):
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child_name, module))
    for parent, child_name, module in replacements:
        _replace_child(parent, child_name, module.merged_linear())
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """仅导出 LoRA 的可训练矩阵，避免重复保存 0.5B 基座参数。"""
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
            if name.endswith("lora_A") or name.endswith("lora_B")}


def save_lora(model: nn.Module, output_dir: str | Path, *, r: int, alpha: float,
              target_modules: Iterable[str]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), output / "adapter_model.pt")
    (output / "adapter_config.json").write_text(
        __import__("json").dumps({"r": r, "alpha": alpha,
                                   "target_modules": list(target_modules)}, indent=2),
        encoding="utf-8",
    )


def load_lora(model: nn.Module, checkpoint_dir: str | Path) -> nn.Module:
    """向已注入相同结构 LoRA 的模型恢复 adapter 权重。"""
    state = torch.load(Path(checkpoint_dir) / "adapter_model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if "lora_" in key]
    if unexpected:
        raise ValueError(f"LoRA checkpoint 含未知参数：{unexpected}")
    if not any(key.endswith("lora_A") for key in state):
        raise ValueError("checkpoint 中没有 LoRA 权重")
    return model
