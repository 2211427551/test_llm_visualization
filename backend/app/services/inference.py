"""模型推理与可视化服务

提供简化的模型初始化与前向推理逻辑，用于与前端进行联调。

- 通过 /initialize 接口初始化模型配置
- 通过 /forward 接口执行模拟推理并生成可视化数据
- 数据格式与前端所需的可视化结构保持一致，便于直接渲染
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

# 默认模型配置，结合 Transformer + MoE 基本参数
DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "modelName": "Transformer-MoE",
    "vocabSize": 32000,
    "contextSize": 256,
    "nLayer": 6,
    "nHead": 8,
    "nEmbed": 512,
    "dropout": 0.1,
    "useSparseAttention": True,
    "useMoe": True,
    "moeNumExperts": 4,
    "moeTopK": 2,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _round(value: float, digits: int = 3) -> float:
    return round(value, digits)


def _generate_heatmap(rows: int, cols: int, phase: float) -> List[List[float]]:
    matrix: List[List[float]] = []
    for r in range(rows):
        row: List[float] = []
        for c in range(cols):
            base = 0.52 + math.sin((r + 1) * 0.42 + phase) * 0.22 + math.cos((c + 1) * 0.31 + phase) * 0.2
            row.append(_round(_clamp(base)))
        matrix.append(row)
    return matrix


def _generate_sparse_matrix(size: int, phase: float, threshold: float) -> List[List[Dict[str, Any]]]:
    matrix: List[List[Dict[str, Any]]] = []
    for r in range(size):
        row: List[Dict[str, Any]] = []
        for c in range(size):
            base = 0.48 + math.exp(-abs(r - c) * 0.28) * 0.4 + math.sin((r + c + 1) * 0.17 + phase) * 0.12
            value = _round(_clamp(base))
            row.append({"value": value, "isSparse": value < threshold})
        matrix.append(row)
    return matrix


def _generate_moe_routes(tokens: List[str], expert_count: int, phase: float) -> Dict[str, Any]:
    limited_tokens = tokens[: min(len(tokens), 8)]
    token_defs = [
        {"id": f"token-{index + 1}", "label": token or f"令牌 {index + 1}"}
        for index, token in enumerate(limited_tokens)
    ]
    experts = [
        {"id": f"expert-{chr(65 + idx)}", "label": f"专家 {chr(65 + idx)}"}
        for idx in range(expert_count)
    ]

    routes: List[Dict[str, Any]] = []
    for token_index, token_def in enumerate(token_defs):
        weights: List[float] = []
        total = 0.0
        for expert_index in range(expert_count):
            base = 0.35 + math.sin((token_index + 1) * 0.45 + expert_index * 0.35 + phase) * 0.25
            value = _clamp(base)
            weights.append(value)
            total += value
        if total == 0:
            total = 1.0
        normalized = [weight / total for weight in weights]
        for expert_index, weight in enumerate(normalized):
            routes.append(
                {
                    "tokenId": token_def["id"],
                    "expertId": experts[expert_index]["id"],
                    "weight": _round(weight, digits=4),
                }
            )
    return {
        "tokens": token_defs,
        "experts": experts,
        "routes": routes,
        "description": "根据门控网络分配权重，动态调用最合适的专家子网络。",
    }


def _tokenize(text: str) -> List[str]:
    # 将中文字符、单词、标点分别作为 token
    pattern = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)
    return [token for token in pattern.findall(text) if token.strip()]


def _build_steps(tokens: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    seq_len = max(len(tokens), 1)
    heatmap_size = min(max(seq_len, 6), 18)
    attention_size = min(max(seq_len, 8), 18)
    moe_expert_count = max(2, min(config.get("moeNumExperts", 4), 8))
    phase = seq_len / 5.0

    embedding_heatmap = _generate_heatmap(heatmap_size, heatmap_size, phase)
    attention_heatmap = _generate_heatmap(attention_size, attention_size, phase + 0.6)
    ff_heatmap = _generate_heatmap(attention_size, attention_size, phase + 1.1)

    sparse_attention = {
        "headLabels": [f"位置 {index + 1}" for index in range(attention_size)],
        "matrix": _generate_sparse_matrix(attention_size, phase + 0.9, threshold=0.18),
        "note": "浅色区域表示被稀疏化的连接，深色表示重点关注的上下文。",
    }

    moe_routing = _generate_moe_routes(tokens, moe_expert_count, phase + 0.3)

    return [
        {
            "id": "step-1",
            "name": "输入解析",
            "description": f"当前输入序列长度 {seq_len}，生成词嵌入与位置编码以构建上下文。",
            "layers": [
                {
                    "id": "layer-1-embedding",
                    "name": "分词嵌入",
                    "type": "input",
                    "summary": "将 token 映射为连续向量空间，保留基础语义距离。",
                    "tensorHeatmap": embedding_heatmap,
                    "sparseAttention": sparse_attention,
                    "moeRouting": moe_routing,
                },
                {
                    "id": "layer-1-position",
                    "name": "位置编码",
                    "type": "embedding",
                    "summary": "利用正弦余弦基函数引入位置信息，与词向量进行融合。",
                    "tensorHeatmap": embedding_heatmap,
                },
                {
                    "id": "layer-1-normalize",
                    "name": "归一化融合",
                    "type": "embedding",
                    "summary": "通过层归一化稳定分布，为多头注意力提供一致的输入。",
                    "tensorHeatmap": embedding_heatmap,
                },
            ],
        },
        {
            "id": "step-2",
            "name": "注意力推理",
            "description": "多头注意力聚合上下文信息，同时结合稀疏策略降低计算开销。",
            "layers": [
                {
                    "id": "layer-2-attention",
                    "name": "多头注意力",
                    "type": "attention",
                    "summary": "关注关键实体关系，突出动宾结构等长程依赖。",
                    "tensorHeatmap": attention_heatmap,
                    "sparseAttention": sparse_attention,
                    "moeRouting": moe_routing,
                },
                {
                    "id": "layer-2-sparse",
                    "name": "稀疏注意力头",
                    "type": "attention",
                    "summary": "稀疏模式仅保留高贡献连接，有效控制推理复杂度。",
                    "tensorHeatmap": attention_heatmap,
                    "sparseAttention": sparse_attention,
                },
                {
                    "id": "layer-2-ffn",
                    "name": "前馈网络",
                    "type": "feedforward",
                    "summary": "逐位置非线性映射，放大注意力提取的关键信号。",
                    "tensorHeatmap": ff_heatmap,
                },
            ],
        },
        {
            "id": "step-3",
            "name": "混合专家聚合",
            "description": "门控网络整合不同专家输出，形成最终的表达与预测。",
            "layers": [
                {
                    "id": "layer-3-gate",
                    "name": "门控决策",
                    "type": "feedforward",
                    "summary": "根据上下文动态选择激活专家，保证表达的多样性。",
                    "tensorHeatmap": ff_heatmap,
                    "moeRouting": moe_routing,
                },
                {
                    "id": "layer-3-combine",
                    "name": "专家聚合",
                    "type": "feedforward",
                    "summary": "按门控权重整合专家输出，生成最终语义表示。",
                    "tensorHeatmap": ff_heatmap,
                },
                {
                    "id": "layer-3-output",
                    "name": "输出投影",
                    "type": "output",
                    "summary": "映射至词汇表空间，准备生成预测文本或分类结果。",
                    "tensorHeatmap": ff_heatmap,
                },
            ],
        },
    ]


@dataclass
class ModelState:
    config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_MODEL_CONFIG.copy())
    initialized_at: Optional[datetime] = None
    last_forward_summary: Optional[Dict[str, Any]] = None


class ModelInferenceService:
    """管理模型初始化与前向推理的服务，内部不依赖真实深度学习模型。"""

    def __init__(self) -> None:
        self._state = ModelState()
        self._lock = Lock()

    def reset(self) -> None:
        """重置模型状态（测试辅助方法）。"""
        with self._lock:
            self._state = ModelState()

    def initialize(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        config = DEFAULT_MODEL_CONFIG.copy()
        if overrides:
            for key, value in overrides.items():
                if key in config and isinstance(value, (int, float, bool)):
                    config[key] = value
        with self._lock:
            self._state.config = config
            self._state.initialized_at = datetime.utcnow()
            self._state.last_forward_summary = None
            initialized_at = self._state.initialized_at
        response = config.copy()
        response["initializedAt"] = initialized_at.isoformat() + "Z"
        return response

    def ensure_initialized(self) -> None:
        if self._state.initialized_at is None:
            raise RuntimeError("模型未初始化，请先调用 /initialize 接口")

    def forward(self, text: str, capture_data: bool) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("输入文本不能为空")

        self.ensure_initialized()
        with self._lock:
            config = self._state.config.copy()

        tokens = _tokenize(text)
        sequence_length = len(tokens)
        if sequence_length == 0:
            raise ValueError("输入文本不能为空")

        if sequence_length > config["contextSize"]:
            raise ValueError(
                f"输入序列长度 {sequence_length} 超过了模型最大长度 {config['contextSize']}"
            )

        logits_shape = [1, sequence_length, config["vocabSize"]]

        captured_data: Optional[Dict[str, Any]] = None
        if capture_data:
            steps = _build_steps(tokens, config)
            runtime = self._build_runtime_summary(sequence_length, logits_shape, config)
            captured_data = {
                "steps": steps,
                "runtime": runtime,
                "tokenSequence": tokens,
                "modelSummary": {
                    "nLayer": config["nLayer"],
                    "nHead": config["nHead"],
                    "nEmbed": config["nEmbed"],
                    "useSparseAttention": config["useSparseAttention"],
                    "useMoe": config["useMoe"],
                },
            }
            with self._lock:
                self._state.last_forward_summary = runtime

        return {
            "logitsShape": logits_shape,
            "sequenceLength": sequence_length,
            "capturedData": captured_data,
        }

    def _build_runtime_summary(
        self, sequence_length: int, logits_shape: List[int], config: Dict[str, Any]
    ) -> Dict[str, Any]:
        complexity = sequence_length * config["nLayer"]
        forward_time_ms = _round(8.5 + complexity * 0.9, digits=2)
        memory_usage = _round(512 + sequence_length * 3.2 + config["nLayer"] * 24, digits=1)
        gpu_util = _round(_clamp(0.35 + sequence_length / (config["contextSize"] * 0.9)), digits=2)

        return {
            "capturedAt": datetime.utcnow().isoformat() + "Z",
            "forwardTimeMs": forward_time_ms,
            "memoryMB": memory_usage,
            "gpuUtilization": gpu_util,
            "batchSize": 1,
            "sequenceLength": sequence_length,
            "logitsShape": logits_shape,
        }


model_inference_service = ModelInferenceService()

__all__ = ["model_inference_service"]
