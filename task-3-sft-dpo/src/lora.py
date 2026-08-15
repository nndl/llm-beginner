"""不依赖 peft 的 LoRA 注入、保存与合并实现。"""
from __future__ import annotations  

from pathlib import Path  
from typing import Iterable  

import torch
from torch import nn
import torch.nn.functional as F  


class LoRALinear(nn.Module):
    """
    LoRA包装层：

    保留冻结的原始 Linear 层：
        y = Wx+b

    增加可训练低秩矩阵：
        ΔW = (alpha/r)BA

    最终输出：
        y = Wx+b+(alpha/r)xA^TB^T

    其中：
        A: 降维矩阵
        B: 升维矩阵
    """

    def __init__(self, base_layer: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        # ===============================
        # 1. 检查LoRA rank
        # ===============================

        # r表示低秩空间维度
        #
        # 原始权重:
        #
        # W:
        # [out_features, in_features]
        #
        # LoRA:
        #
        # B:
        # [out_features,r]
        #
        # A:
        # [r,in_features]
        #
        # r必须 > 0
        if r <= 0:  # LoRA 的秩必须为正，否则 A/B 矩阵无意义
            raise ValueError("LoRA rank r 必须为正数")
        # ===============================
        # 2. 保存原始Linear层
        # ===============================


        # 原始模型权重:
        #
        # W
        #
        # 保留用于forward计算
        #
        # 但是后面冻结，不参与训练
        self.base_layer = base_layer  # 保留原始 Linear，前向时仍参与计算（只是不再更新梯度）
        self.r = r  # 低秩分解的秩，控制新增参数量
        self.alpha = alpha  # LoRA 论文中的缩放系数分子，用于控制低秩更新的影响强度
        # 缩放因子
        #
        # ΔW = alpha/r * BA
        #
        # 例如：
        #
        # r=8
        # alpha=16
        #
        # scaling=2
        self.scaling = alpha / r  # 实际叠加时的缩放因子：alpha 越大或 r 越小，低秩更新影响越大
        # ===============================
        # 3. 创建LoRA参数
        # ===============================


        # 保证LoRA参数：
        #
        # 和原模型:
        #
        # GPU位置一致
        # dtype一致
        #
        # 例如：
        # CUDA + FP16
        #
        # 不会产生device mismatch
        factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}  # 保证新参数与基座同设备同精度
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features, **factory_kwargs))  # 降维矩阵 A：(r, in_features)
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, r, **factory_kwargs))  # 升维矩阵 B：(out_features, r)
        # ===============================
        # 4. 初始化LoRA参数
        # ===============================
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)  # A 用 kaiming 初始化产生非零梯度信号
        # B 全零初始化（对应上面 torch.zeros）：训练开始时 B @ A = 0，LoRA 分支不改变基座原始输出
        # ===============================
        # 5. 冻结基础模型
        # ===============================
        for parameter in self.base_layer.parameters():  # 遍历基座 Linear 的 weight/bias
            parameter.requires_grad = False  # 冻结基座参数，只训练新增的 lora_A / lora_B
    # ==================================
    # 6. 保持Linear接口兼容
    # ==================================
    @property
    def in_features(self) -> int:
        return self.base_layer.in_features  # 直接转发基座层的输入维度，便于外部代码按 nn.Linear 接口使用

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features  # 直接转发基座层的输出维度

    # ==================================
    # 7. 前向传播
    # ==================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)  # 原始（冻结）Linear 的输出
        # LoRA分支:
        #
        # 第一步:
        #
        # xA^T
        #
        # 降维:
        #
        # [batch,in]
        #
        # ->
        #
        # [batch,r]
        #
        #
        # 第二步:
        #
        # B
        #
        # 升维:
        #
        # [batch,r]
        #
        # ->
        #
        # [batch,out]
        #
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)  # 先降维到 r 维（x @ A^T），再升维回 out_features（· @ B^T）
        return base + update * self.scaling  # 基座输出叠加按 alpha/r 缩放后的低秩更新
    # ==================================
    # 8. 权重融合
    # ==================================
    @torch.no_grad()  # 合并权重是纯算术操作，不需要构建计算图
    def merged_linear(self) -> nn.Linear:
        """
        将LoRA权重融合回原Linear。

        推理时：

        原来:

        Linear(x)+LoRA(x)


        合并后:

        Linear'(x)


        速度等同原模型。
        """
        base = self.base_layer
        merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                           device=base.weight.device, dtype=base.weight.dtype)  # 新建一个形状/设备/精度都对齐的普通 Linear
                           # 权重融合:
        #
        # W'=W+alpha/r*BA
        #
        merged.weight.copy_(base.weight + (self.lora_B @ self.lora_A).to(base.weight.dtype) * self.scaling)  # 把 B@A（即低秩更新的等价权重）按同样缩放叠加进原始权重
        if base.bias is not None:
            merged.bias.copy_(base.bias)  # bias 不受 LoRA 影响，直接复制
        return merged  # 返回的普通 Linear 推理时无需再计算 LoRA 分支，速度和显存与原模型一致


def _replace_child(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):  # 命名形如 "0"/"1" 且父模块是按下标索引的容器
        parent[int(child_name)] = child  # 用整数下标赋值替换该子模块
    else:
        setattr(parent, child_name, child)  # 普通具名子模块，直接按属性名替换


def inject_lora(model: nn.Module, target_modules: Iterable[str], r: int = 8,
                alpha: float = 16.0) -> nn.Module:
    """
    给指定 Linear 层注入 LoRA。

    参数：
        model:
            原始大模型，例如 Qwen/LLaMA

        target_modules:
            需要注入LoRA的模块名字

            例如：
            [
                "q_proj",
                "v_proj"
            ]

        r:
            LoRA低秩维度

        alpha:
            LoRA缩放系数


    操作流程：

        1. 冻结整个模型
        2. 找到目标Linear层
        3. 替换成LoRALinear
        4. 返回修改后的模型
    """
    # ======================================
    # 1. 保存目标模块名称
    # ======================================


    # 转换为set
    #
    # 原:
    #
    # ["q_proj","v_proj"]
    #
    # 变:
    #
    # {"q_proj","v_proj"}
    #
    # 查询复杂度:
    #
    # list:
    # O(n)
    #
    # set:
    # O(1)
    #
    targets = set(target_modules)  # 转成集合便于 O(1) 成员判断
    # ======================================
    # 2. 冻结整个基础模型
    # ======================================


    # 遍历所有参数
    #
    # 包括:
    #
    # embedding
    # attention
    # FFN
    # norm
    #
    for parameter in model.parameters():  # 先冻结整个模型的全部参数
        parameter.requires_grad = False  # 关闭梯度计算，后续只有新注入的 LoRA 矩阵会被设为可训练

    # ======================================
    # 3. 搜索需要替换的Linear层
    # ======================================


    # 保存待替换列表
    #
    # 为什么不直接替换？
    #
    # 因为:
    #
    # named_modules()
    # 遍历过程中修改模型结构
    #
    # 会导致:
    #
    # RuntimeError:
    # dictionary changed size during iteration
    #
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []  # 先收集待替换项，避免遍历时修改模型结构
    # 遍历所有子模块
    #
    # 返回:
    #
    # full_name:
    # 完整路径
    #
    # module:
    # 模块对象
    #
    for full_name, module in model.named_modules():  # 遍历模型中所有子模块及其完整点分路径名
        # 跳过根节点
        #
        # 根:
        #
        # full_name=""
        #
        # model本身
        #
        #
        # 同时跳过非Linear层
        if not full_name or not isinstance(module, nn.Linear):  # 跳过根模块本身（full_name==""）和非 Linear 层
            continue
        # 获取当前Linear最后一级名字
        #
        # 例如:
        #
        # model.layers.0.self_attn.q_proj
        #
        #
        # rsplit:
        #
        # [
        # "model.layers.0.self_attn",
        # "q_proj"
        # ]
        #
        #
        # 取:
        #
        # q_proj
        #
        # 判断是否是目标模块
        #
        # 例如:
        #
        # q_proj
        # v_proj
        #
        if full_name.rsplit(".", 1)[-1] not in targets:  # 只匹配模块名的最后一段（如 "q_proj"），忽略路径前缀
            continue
        # ==================================
        # 4. 分离父模块和子模块名字
        # ==================================


        # 举例:
        #
        # full_name:
        #
        # model.layers.0.self_attn.q_proj
        #
        #
        # 分割:
        #
        # parent_name:
        #
        # model.layers.0.self_attn
        #
        #
        # child_name:
        #
        # q_proj
        parent_name, _, child_name = full_name.rpartition(".")  # 拆出父模块路径与该子模块自己的名字
        # 根据路径找到父模块
        #
        # 例如:
        #
        # model.layers.0.self_attn
        #
        parent = model.get_submodule(parent_name) if parent_name else model  # 顶层模块的父模块就是 model 本身
        replacements.append((parent, child_name, module))  # 记录：父模块、子模块名、待替换的原始 Linear
    # ======================================
    # 5. 检查是否找到目标层
    # ======================================
    if not replacements:  # 一个都没匹配到，说明 target_modules 拼写有误或模型结构不含这些层
        raise ValueError(f"未找到 LoRA 目标 Linear 层：{sorted(targets)}")
    # ======================================
    # 6. 执行模块替换
    # ======================================
    for parent, child_name, module in replacements:  # 统一在收集阶段结束后再执行替换
        _replace_child(parent, child_name, LoRALinear(module, r=r, alpha=alpha))  # 用包装了原始 Linear 的 LoRALinear 替换
    return model  # 原地修改后返回同一个 model 对象，便于链式调用


def merge_lora(model: nn.Module) -> nn.Module:
    """
    将模型中所有 LoRA 更新合并到基础 Linear 权重中，
    并将 LoRALinear 替换回普通 nn.Linear。

    合并前：

        Linear + LoRA

        y = Wx + (alpha / r) * BAx

    合并后：

        Linear

        y = W' x

    其中：

        W' = W + (alpha / r) * BA
    """
    # ==========================================
    # 1. 保存所有需要合并的 LoRALinear
    # ==========================================

    # 不直接在 named_modules() 遍历过程中修改模型结构，
    # 而是先把需要替换的模块保存下来。
    #
    # 每个元素包含：
    #
    # parent：
    #     LoRALinear 所属的父模块
    #
    # child_name：
    #     LoRALinear 在父模块中的名字
    #
    # module：
    #     当前 LoRALinear 对象
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []  # 同样先收集，避免遍历时改结构
    # ==========================================
    # 2. 遍历模型中的所有子模块
    # ==========================================

    # named_modules() 会返回：
    #
    # full_name:
    #     子模块的完整路径
    #
    # module:
    #     子模块对象
    #
    # 例如：
    #
    # model.layers.0.self_attn.q_proj
    #
    # 对应：
    #
    # full_name =
    # "model.layers.0.self_attn.q_proj"
    #
    # module =
    # LoRALinear(...)
    for full_name, module in model.named_modules():  # 找出模型中所有已注入的 LoRALinear
        if not full_name or not isinstance(module, LoRALinear):
            continue
        # ==========================================
        # 3. 获取父模块和当前模块名称
        # ==========================================
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child_name, module))   # 保存待替换信息
    # ==========================================
    # 4. 执行LoRA权重合并
    # ==========================================
    # module.merged_linear()
        #
        # 会执行：
        #
        # W_new =
        # W + (alpha / r) * B @ A
        #
        # 并返回一个普通 nn.Linear。
        #
        # 然后使用 _replace_child()
        # 将原来的 LoRALinear 替换掉。
    for parent, child_name, module in replacements:
        _replace_child(parent, child_name, module.merged_linear())  # 替换回不含 LoRA 分支的普通 Linear
    return model # 返回已经完成LoRA合并的模型


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """仅导出 LoRA 的可训练矩阵，避免重复保存 0.5B 基座参数。"""
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
            if name.endswith("lora_A") or name.endswith("lora_B")}  # 按参数名后缀筛出 LoRA 矩阵，并搬到 CPU 便于保存


def save_lora(model: nn.Module, output_dir: str | Path, *, r: int, alpha: float,
              target_modules: Iterable[str]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)  # 输出目录不存在时自动创建（含多级父目录）
    torch.save(lora_state_dict(model), output / "adapter_model.pt")  # 只保存 LoRA 矩阵，体积远小于整模型
    (output / "adapter_config.json").write_text(
        __import__("json").dumps({"r": r, "alpha": alpha,
                                   "target_modules": list(target_modules)}, indent=2),
        encoding="utf-8",
    )  # 同时保存超参数，供 load_lora 前重新 inject_lora 时对齐结构


def load_lora(model: nn.Module, checkpoint_dir: str | Path) -> nn.Module:
    """向已注入相同结构 LoRA 的模型恢复 adapter 权重。"""
    state = torch.load(Path(checkpoint_dir) / "adapter_model.pt", map_location="cpu", weights_only=True)  # 只反序列化张量数据，不执行任意代码，规避反序列化风险
    missing, unexpected = model.load_state_dict(state, strict=False)  # 非严格加载：允许 state 中缺少基座参数（本就没保存）
    unexpected = [key for key in unexpected if "lora_" in key]  # 在“意外的键”里只关心 LoRA 相关的，排除基座层名字的误报
    if unexpected:  # 如果 checkpoint 里有加载失败的 LoRA 键，说明模型结构和 checkpoint 不匹配
        raise ValueError(f"LoRA checkpoint 含未知参数：{unexpected}")
    if not any(key.endswith("lora_A") for key in state):  # 兜底校验：checkpoint 至少要含一个 lora_A，否则可能加载错文件
        raise ValueError("checkpoint 中没有 LoRA 权重")
    return model
