"""
前向数据捕获模式定义

定义了用于序列化模型前向传播过程中所有中间数据的Pydantic模式。
所有字段名和注释都使用简体中文，确保数据结构的可JSON化特性。

主要包含：
- 嵌入层数据：词嵌入和位置编码
- 稀疏注意力数据：各层注意力模式和权重
- MoE路由数据：专家选择和路由信息
- 最终输出数据：logits和汇总统计
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime

try:
    from pydantic import BaseModel, Field, ConfigDict
    
    class 嵌入数据(BaseModel):
        """嵌入层数据捕获模式
        
        记录词嵌入和位置编码的详细信息
        """
        model_config = ConfigDict(
            title="嵌入数据",
            description="嵌入层输出的词嵌入和位置编码信息",
            use_enum_values=True
        )
        
        输入序列长度: int = Field(..., description="输入token序列的长度")
        批次大小: int = Field(..., description="批次大小")
        嵌入维度: int = Field(..., description="嵌入向量维度")
        词嵌入形状: List[int] = Field(..., description="词嵌入张量的形状")
        位置编码形状: List[int] = Field(..., description="位置编码张量的形状")
        融合嵌入形状: List[int] = Field(..., description="融合后嵌入张量的形状")
        词汇表大小: int = Field(..., description="词汇表总大小")
        最大位置: int = Field(..., description="最大位置编码索引")


    class 注意力头信息(BaseModel):
        """单个注意力头的详细信息"""
        
        头索引: int = Field(..., description="注意力头的索引")
        注意力类型: str = Field(..., description="注意力类型：局部/全局/标准")
        查询形状: List[int] = Field(..., description="查询张量形状")
        键形状: List[int] = Field(..., description="键张量形状")
        值形状: List[int] = Field(..., description="值张量形状")
        注意力权重形状: List[int] = Field(..., description="注意力权重张量形状")
        输出形状: List[int] = Field(..., description="注意力输出张量形状")
        窗口大小: Optional[int] = Field(None, description="局部注意力窗口大小")
        全局token数量: Optional[int] = Field(None, description="全局注意力token数量")


    class 稀疏注意力数据(BaseModel):
        """稀疏注意力层数据捕获模式
        
        记录每层稀疏注意力的详细信息和模式
        """
        model_config = ConfigDict(
            title="稀疏注意力数据",
            description="稀疏注意力层的计算模式和权重信息"
        )
        
        层索引: int = Field(..., description="Transformer层的索引")
        注意力类型: str = Field(..., description="注意力机制类型")
        注意力头数量: int = Field(..., description="总注意力头数量")
        局部注意力头数: int = Field(..., description="局部注意力头数量")
        全局注意力头数: int = Field(..., description="全局注意力头数量")
        注意力头信息: List[注意力头信息] = Field(..., description="各注意力头的详细信息")
        输入序列长度: int = Field(..., description="输入序列长度")
        动态窗口大小: Optional[int] = Field(None, description="实际使用的动态窗口大小")
        计算复杂度: Optional[str] = Field(None, description="计算复杂度描述")


    class MoE专家信息(BaseModel):
        """单个MoE专家的处理信息"""
        
        专家索引: int = Field(..., description="专家网络的索引")
        处理token数量: int = Field(..., description="该专家处理的token数量")
        平均权重: float = Field(..., description="该专家的平均路由权重")
        最大权重: float = Field(..., description="该专家的最大路由权重")
        输入形状: List[int] = Field(..., description="专家输入张量形状")
        输出形状: List[int] = Field(..., description="专家输出张量形状")
        激活函数: str = Field(..., description="专家使用的激活函数")


    class MoE路由数据(BaseModel):
        """MoE层数据捕获模式
        
        记录MoE路由选择和专家处理的详细信息
        """
        model_config = ConfigDict(
            title="MoE路由数据",
            description="Mixture of Experts层的路由和专家处理信息"
        )
        
        层索引: int = Field(..., description="Transformer层的索引")
        专家总数: int = Field(..., description="MoE专家总数")
        TopK值: int = Field(..., description="每个token选择的专家数量")
        输入token总数: int = Field(..., description="输入的token总数")
        门控分数形状: List[int] = Field(..., description="门控网络输出分数的形状")
        专家信息列表: List[MoE专家信息] = Field(..., description="各专家的详细处理信息")
        负载均衡损失: float = Field(..., description="负载均衡损失值")
        专家使用标准差: float = Field(..., description="专家使用频率的标准差")
        路由确定性: float = Field(..., description="路由决策的确定性指标")


    class Transformer层数据(BaseModel):
        """单个Transformer层的完整数据"""
        
        层索引: int = Field(..., description="Transformer层的索引")
        输入形状: List[int] = Field(..., description="层输入张量形状")
        输出形状: List[int] = Field(..., description="层输出张量形状")
        注意力数据: Optional[稀疏注意力数据] = Field(None, description="稀疏注意力数据")
        MoE数据: Optional[MoE路由数据] = Field(None, description="MoE路由数据")
        层类型: str = Field(..., description="层类型：标准/稀疏注意力/MoE")


    class 最终输出数据(BaseModel):
        """模型最终输出数据"""
        
        Logits形状: List[int] = Field(..., description="最终输出logits张量形状")
        词汇表大小: int = Field(..., description="输出词汇表大小")
        最大logits值: float = Field(..., description="logits中的最大值")
        最小logits值: float = Field(..., description="logits中的最小值")
        平均logits值: float = Field(..., description="logits的平均值")
        输出概率熵: Optional[float] = Field(None, description="输出概率的熵值")


    class 前向传播完整轨迹(BaseModel):
        """前向传播完整轨迹数据捕获模式
        
        包含从输入到输出的所有中间数据，用于完整的模型行为分析
        """
        model_config = ConfigDict(
            title="前向传播完整轨迹",
            description="模型前向传播过程中所有中间数据的完整记录"
        )
        
        捕获时间: datetime = Field(default_factory=datetime.now, description="数据捕获的时间戳")
        模型配置: Dict[str, Any] = Field(..., description="模型配置信息")
        
        # 输入信息
        输入ID形状: List[int] = Field(..., description="输入token ID张量的形状")
        批次大小: int = Field(..., description="处理批次大小")
        序列长度: int = Field(..., description="输入序列长度")
        
        # 嵌入层数据
        嵌入数据: 嵌入数据 = Field(..., description="嵌入层的详细数据")
        
        # 各层Transformer数据
        Transformer层数据: List[Transformer层数据] = Field(..., description="各Transformer层的详细数据")
        
        # 最终输出
        最终输出: 最终输出数据 = Field(..., description="模型最终输出数据")
        
        # 性能统计
        总参数数量: int = Field(..., description="模型总参数数量")
        可训练参数数量: int = Field(..., description="可训练参数数量")
        捕耗时毫秒: Optional[float] = Field(None, description="数据捕获耗时（毫秒）")
        
        # 元数据
        捕获模式: str = Field(default="完整", description="数据捕获模式：完整/简化/自定义")
        设备类型: str = Field(..., description="计算设备类型：CPU/GPU")
        数据类型: str = Field(..., description="张量数据类型：float32/float16等")


    class 数据捕获配置(BaseModel):
        """数据捕获配置模式
        
        定义数据捕获的详细配置选项
        """
        model_config = ConfigDict(
            title="数据捕获配置",
            description="控制数据捕获行为的配置参数"
        )
        
        捕获嵌入数据: bool = Field(default=True, description="是否捕获嵌入层数据")
        捕获注意力数据: bool = Field(default=True, description="是否捕获注意力数据")
        捕获MoE数据: bool = Field(default=True, description="是否捕获MoE路由数据")
        捕获最终输出: bool = Field(default=True, description="是否捕获最终输出数据")
        
        捕获张量值: bool = Field(default=False, description="是否捕获实际张量值（可能消耗大量内存）")
        捕获统计信息: bool = Field(default=True, description="是否捕获张量统计信息")
        
        最大捕获层数: Optional[int] = Field(None, description="最大捕获的层数，None表示全部")
        采样率: float = Field(default=1.0, description="数据采样率，1.0表示全部捕获")
        
        内存限制MB: Optional[int] = Field(None, description="内存使用限制（MB）")
        压缩数据: bool = Field(default=False, description="是否压缩捕获的数据")

except ImportError:
    # Fallback for testing without pydantic
    print("Warning: Pydantic not available, using fallback classes")
    
    class 嵌入数据:
        pass
    
    class 注意力头信息:
        pass
    
    class 稀疏注意力数据:
        pass
    
    class MoE专家信息:
        pass
    
    class MoE路由数据:
        pass
    
    class Transformer层数据:
        pass
    
    class 最终输出数据:
        pass
    
    class 前向传播完整轨迹:
        pass
    
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