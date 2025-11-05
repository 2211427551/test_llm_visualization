"""
前向数据捕获容器

实现模型前向传播过程中的数据收集容器，负责：
- 安全地捕获embedding、稀疏注意力、MoE路由等中间数据
- 使用torch.no_grad()和detach()避免影响梯度计算
- 将数据转换为可JSON化的Pydantic结构
- 提供内存管理和性能优化功能

设计原则：
1. 非侵入性：捕获逻辑不干扰正常的模型计算
2. 高效性：最小化性能开销和内存占用
3. 完整性：捕获所有关键的前向传播信息
4. 可配置性：支持灵活的捕获策略配置
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass

try:
    from ..schemas.forward_capture import (
        嵌入数据, 稀疏注意力数据, MoE路由数据, Transformer层数据,
        最终输出数据, 前向传播完整轨迹, 数据捕获配置,
        注意力头信息, MoE专家信息
    )
    from ..models.transformer.config import GPT2Config
except ImportError:
    # Fallback for testing without pydantic
    print("Warning: Using fallback schemas for testing")
    
    class 数据捕获配置:
        def __init__(self, **kwargs):
            self.捕获嵌入数据 = kwargs.get('捕获嵌入数据', True)
            self.捕获注意力数据 = kwargs.get('捕获注意力数据', True)
            self.捕获MoE数据 = kwargs.get('捕获MoE数据', True)
            self.捕获最终输出 = kwargs.get('捕获最终输出', True)
            self.捕获张量值 = kwargs.get('捕获张量值', False)
            self.捕获统计信息 = kwargs.get('捕获统计信息', True)
            self.最大捕获层数 = kwargs.get('最大捕获层数', None)
            self.内存限制MB = kwargs.get('内存限制MB', None)
    
    class GPT2Config:
        def __init__(self, **kwargs):
            self.vocab_size = kwargs.get('vocab_size', 1000)
            self.n_embed = kwargs.get('n_embed', 256)
            self.context_size = kwargs.get('context_size', 256)
            self.moe_activation = kwargs.get('moe_activation', 'gelu')
            self.moe_num_experts = kwargs.get('moe_num_experts', 4)
            self.moe_top_k = kwargs.get('moe_top_k', 2)


class 数据捕获容器:
    """前向数据捕获容器
    
    负责在模型前向传播过程中安全地收集所有中间数据。
    使用torch.no_grad()确保捕获过程不影响梯度计算。
    """
    
    def __init__(self, config, capture_config=None):
        """
        初始化数据捕获容器
        
        Args:
            config: 模型配置对象
            capture_config: 数据捕获配置，如果为None则使用默认配置
        """
        self.model_config = config
        self.capture_config = capture_config or 数据捕获配置()
        
        # 捕获状态
        self.is_capturing = False
        self.capture_start_time = None
        
        # 数据存储
        self.嵌入数据缓存 = None
        self.层数据列表 = []
        self.最终输出缓存 = None
        
        # 性能统计
        self.memory_usage = 0
        self.capture_times = []
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def 捕获上下文(self):
        """数据捕获上下文管理器
        
        确保捕获过程的正确初始化和清理
        """
        if self.is_capturing:
            raise RuntimeError("数据捕获已在进行中")
        
        self.is_capturing = True
        self.capture_start_time = time.time()
        self.嵌入数据缓存 = None
        self.层数据列表.clear()
        self.最终输出缓存 = None
        
        try:
            with torch.no_grad():  # 确保不影响梯度
                yield self
        finally:
            self.is_capturing = False
            if self.capture_start_time:
                capture_duration = (time.time() - self.capture_start_time) * 1000
                self.capture_times.append(capture_duration)
    
    def 捕获嵌入数据(self, 
                      input_ids, 
                      token_embeddings,
                      position_embeddings,
                      final_embeddings) -> None:
        """捕获嵌入层数据"""
        if not self.is_capturing or not self.capture_config.捕获嵌入数据:
            return
        
        try:
            with torch.no_grad():
                batch_size, seq_len = input_ids.shape
                
                # 分离张量以避免梯度影响
                input_ids_detached = input_ids.detach()
                token_emb_detached = token_embeddings.detach()
                pos_emb_detached = position_embeddings.detach()
                final_emb_detached = final_embeddings.detach()
                
                self.嵌入数据缓存 = {
                    "输入序列长度": seq_len,
                    "批次大小": batch_size,
                    "嵌入维度": self.model_config.n_embed,
                    "词嵌入形状": list(token_emb_detached.shape),
                    "位置编码形状": list(pos_emb_detached.shape),
                    "融合嵌入形状": list(final_emb_detached.shape),
                    "词汇表大小": self.model_config.vocab_size,
                    "最大位置": self.model_config.context_size,
                }
                
                # 更新内存使用统计
                self._update_memory_usage(final_emb_detached)
                
        except Exception as e:
            self.logger.error(f"捕获嵌入数据时出错: {e}")
    
    def 捕获稀疏注意力数据(self, 
                           layer_idx: int,
                           attention_type: str,
                           attention_output,
                           intermediate_data=None) -> None:
        """捕获稀疏注意力数据"""
        if not self.is_capturing or not self.capture_config.捕获注意力数据:
            return
        
        # 检查层数限制
        if (self.capture_config.最大捕获层数 is not None and 
            layer_idx >= self.capture_config.最大捕获层数):
            return
        
        try:
            with torch.no_grad():
                attention_output_detached = attention_output.detach()
                
                layer_data = {
                    "层索引": layer_idx,
                    "层类型": "稀疏注意力",
                    "输入形状": list(attention_output_detached.shape),
                    "输出形状": list(attention_output_detached.shape),
                    "注意力类型": attention_type,
                }
                
                # 如果有中间数据，提取详细信息
                if intermediate_data:
                    layer_data.update(self._提取注意力中间数据(intermediate_data))
                
                self.层数据列表.append(layer_data)
                self._update_memory_usage(attention_output_detached)
                
        except Exception as e:
            self.logger.error(f"捕获稀疏注意力数据时出错: {e}")
    
    def 捕获MoE路由数据(self, 
                        layer_idx: int,
                        moe_output,
                        intermediate_data=None) -> None:
        """捕获MoE路由数据"""
        if not self.is_capturing or not self.capture_config.捕获MoE数据:
            return
        
        # 检查层数限制
        if (self.capture_config.最大捕获层数 is not None and 
            layer_idx >= self.capture_config.最大捕获层数):
            return
        
        try:
            with torch.no_grad():
                moe_output_detached = moe_output.detach()
                
                layer_data = {
                    "层索引": layer_idx,
                    "层类型": "MoE",
                    "输入形状": list(moe_output_detached.shape),
                    "输出形状": list(moe_output_detached.shape),
                }
                
                # 如果有中间数据，提取MoE详细信息
                if intermediate_data:
                    layer_data.update(self._提取MoE中间数据(intermediate_data))
                
                self.层数据列表.append(layer_data)
                self._update_memory_usage(moe_output_detached)
                
        except Exception as e:
            self.logger.error(f"捕获MoE路由数据时出错: {e}")
    
    def 捕获标准层数据(self, 
                       layer_idx: int,
                       layer_output,
                       layer_type: str = "标准") -> None:
        """捕获标准Transformer层数据"""
        if not self.is_capturing:
            return
        
        # 检查层数限制
        if (self.capture_config.最大捕获层数 is not None and 
            layer_idx >= self.capture_config.最大捕获层数):
            return
        
        try:
            with torch.no_grad():
                layer_output_detached = layer_output.detach()
                
                layer_data = {
                    "层索引": layer_idx,
                    "层类型": layer_type,
                    "输入形状": list(layer_output_detached.shape),
                    "输出形状": list(layer_output_detached.shape),
                }
                
                self.层数据列表.append(layer_data)
                self._update_memory_usage(layer_output_detached)
                
        except Exception as e:
            self.logger.error(f"捕获标准层数据时出错: {e}")
    
    def 捕获最终输出(self, logits) -> None:
        """捕获最终输出数据"""
        if not self.is_capturing or not self.capture_config.捕获最终输出:
            return
        
        try:
            with torch.no_grad():
                logits_detached = logits.detach()
                
                # 计算统计信息
                if self.capture_config.捕获统计信息:
                    logits_cpu = logits_detached.cpu()
                    max_val = float(logits_cpu.max().item())
                    min_val = float(logits_cpu.min().item())
                    mean_val = float(logits_cpu.mean().item())
                else:
                    max_val = min_val = mean_val = None
                
                self.最终输出缓存 = {
                    "Logits形状": list(logits_detached.shape),
                    "词汇表大小": self.model_config.vocab_size,
                    "最大logits值": max_val,
                    "最小logits值": min_val,
                    "平均logits值": mean_val,
                }
                
                self._update_memory_usage(logits_detached)
                
        except Exception as e:
            self.logger.error(f"捕获最终输出时出错: {e}")
    
    def 生成完整轨迹(self, model, input_ids):
        """生成完整的前向传播轨迹"""
        if not self.is_capturing:
            raise RuntimeError("必须在捕获上下文中调用此方法")
        
        try:
            # 计算捕获耗时
            capture_time_ms = None
            if self.capture_start_time:
                capture_time_ms = (time.time() - self.capture_start_time) * 1000
            
            # 获取模型参数统计
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 构建模型配置字典
            model_config_dict = {
                "vocab_size": getattr(self.model_config, 'vocab_size', 1000),
                "context_size": getattr(self.model_config, 'context_size', 256),
                "n_layer": getattr(self.model_config, 'n_layer', 4),
                "n_head": getattr(self.model_config, 'n_head', 8),
                "n_embed": getattr(self.model_config, 'n_embed', 256),
                "use_sparse_attention": getattr(self.model_config, 'use_sparse_attention', False),
                "use_moe": getattr(self.model_config, 'use_moe', False),
                "moe_num_experts": getattr(self.model_config, 'moe_num_experts', 4),
                "moe_top_k": getattr(self.model_config, 'moe_top_k', 2),
            }
            
            # 构建轨迹数据
            trajectory = {
                "模型配置": model_config_dict,
                "输入ID形状": list(input_ids.shape),
                "批次大小": input_ids.shape[0],
                "序列长度": input_ids.shape[1],
                "嵌入数据": self.嵌入数据缓存,
                "Transformer层数据": self.层数据列表,
                "最终输出": self.最终输出缓存,
                "总参数数量": total_params,
                "可训练参数数量": trainable_params,
                "捕耗时毫秒": capture_time_ms,
                "捕获模式": "完整" if self.capture_config.捕获张量值 else "简化",
                "设备类型": str(input_ids.device).split(":")[0],
                "数据类型": str(input_ids.dtype),
            }
            
            return trajectory
            
        except Exception as e:
            self.logger.error(f"生成完整轨迹时出错: {e}")
            raise
    
    def _提取注意力中间数据(self, intermediate_data):
        """提取注意力中间数据的详细信息"""
        extracted = {}
        
        try:
            # 提取稀疏注意力特定信息
            extracted.update({
                "注意力头数量": intermediate_data.get("num_heads", 0),
                "局部注意力头数": intermediate_data.get("local_heads", 0),
                "全局注意力头数": intermediate_data.get("global_heads", 0),
                "输入序列长度": intermediate_data.get("seq_len", 0),
                "动态窗口大小": intermediate_data.get("dynamic_window_size"),
                "计算复杂度": intermediate_data.get("complexity"),
            })
            
        except Exception as e:
            self.logger.error(f"提取注意力中间数据时出错: {e}")
        
        return extracted
    
    def _提取MoE中间数据(self, intermediate_data):
        """提取MoE中间数据的详细信息"""
        extracted = {}
        
        try:
            # 计算路由确定性
            gate_scores = intermediate_data.get("gate_scores", torch.zeros(1))
            routing_certainty = 0.0
            if gate_scores.numel() > 0:
                # 使用熵的负值作为确定性指标
                try:
                    entropy = -(gate_scores * torch.log(gate_scores + 1e-8)).sum(dim=-1)
                    routing_certainty = float(entropy.mean().item())
                except:
                    routing_certainty = 0.0
            
            extracted.update({
                "专家总数": intermediate_data.get("num_experts", getattr(self.model_config, 'moe_num_experts', 4)),
                "TopK值": intermediate_data.get("top_k", getattr(self.model_config, 'moe_top_k', 2)),
                "输入token总数": intermediate_data.get("total_tokens", 0),
                "门控分数形状": list(intermediate_data.get("gate_scores", torch.zeros(1)).shape),
                "负载均衡损失": float(intermediate_data.get("load_balance_loss", 0.0)),
                "专家使用标准差": float(intermediate_data.get("expert_usage_std", 0.0)),
                "路由确定性": routing_certainty,
            })
            
        except Exception as e:
            self.logger.error(f"提取MoE中间数据时出错: {e}")
        
        return extracted
    
    def _update_memory_usage(self, tensor):
        """更新内存使用统计"""
        if self.capture_config.内存限制MB:
            # 计算张量内存占用（字节）
            tensor_memory = tensor.numel() * tensor.element_size()
            self.memory_usage += tensor_memory
            
            # 检查是否超过内存限制
            limit_bytes = self.capture_config.内存限制MB * 1024 * 1024
            if self.memory_usage > limit_bytes:
                self.logger.warning(
                    f"内存使用超过限制: {self.memory_usage / 1024 / 1024:.2f}MB > {self.capture_config.内存限制MB}MB"
                )
    
    def 获取性能统计(self):
        """获取捕获性能统计信息"""
        return {
            "平均捕获时间": sum(self.capture_times) / len(self.capture_times) if self.capture_times else 0,
            "总捕获次数": len(self.capture_times),
            "内存使用MB": self.memory_usage / 1024 / 1024,
            "捕获层数": len(self.层数据列表),
        }