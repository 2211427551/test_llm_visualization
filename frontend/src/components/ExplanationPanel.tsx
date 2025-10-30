'use client';

import { useVisualizationStore } from '@/store/visualizationStore';
import { Card } from '@/components/ui/card';
import { BookOpen, ExternalLink, Video } from 'lucide-react';

export default function ExplanationPanel() {
  const { isInitialized, currentStepData, currentStep } = useVisualizationStore();

  if (!isInitialized) {
    return (
      <Card className="sticky top-24">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">详细解释</h3>
        </div>
        <p className="text-slate-400">等待计算初始化...</p>
      </Card>
    );
  }

  const getDetailedExplanation = (stepType: string): { title: string; content: string; points: string[] } => {
    const explanations: Record<string, { title: string; content: string; points: string[] }> = {
      'layer_norm_1': {
        title: 'Layer Normalization (注意力前)',
        content: '在多头注意力机制之前，对输入进行Layer Normalization。',
        points: [
          '目的: 稳定训练，加速收敛',
          '操作: 对每个样本的特征维度进行归一化',
          '公式: LN(x) = γ * (x - μ) / σ + β'
        ]
      },
      'q_projection': {
        title: 'Query 投影',
        content: '将输入通过线性变换生成Query矩阵。',
        points: [
          '操作: Q = X @ W_q',
          'Query用于表示"查询"信息',
          '维度: [序列长度, n_embd]'
        ]
      },
      'k_projection': {
        title: 'Key 投影',
        content: '将输入通过线性变换生成Key矩阵。',
        points: [
          '操作: K = X @ W_k',
          'Key用于被Query查询匹配',
          '维度: [序列长度, n_embd]'
        ]
      },
      'v_projection': {
        title: 'Value 投影',
        content: '将输入通过线性变换生成Value矩阵。',
        points: [
          '操作: V = X @ W_v',
          'Value存储实际要传递的信息',
          '维度: [序列长度, n_embd]'
        ]
      },
      'attention_scores': {
        title: '注意力分数计算',
        content: '计算Query和Key之间的相似度。',
        points: [
          '操作: scores = (Q @ K^T) / sqrt(d_k)',
          '衡量不同位置之间的相关性',
          '缩放: 除以sqrt(d_k)防止梯度消失'
        ]
      },
      'attention_weights': {
        title: '注意力权重 (Softmax)',
        content: '对注意力分数应用Softmax，得到归一化的权重。',
        points: [
          '操作: weights = softmax(scores)',
          '将分数转换为概率分布',
          '特性: 所有权重和为1，范围[0, 1]'
        ]
      },
      'attention_output': {
        title: '注意力输出',
        content: '使用注意力权重对Value进行加权求和。',
        points: [
          '操作: output = weights @ V',
          '聚合相关信息',
          '结果: 每个位置获得上下文信息'
        ]
      },
      'moe_router': {
        title: 'MoE 路由器',
        content: '决定每个token应该被哪些专家处理。',
        points: [
          '计算每个token对每个专家的路由分数',
          '选择top-k个专家处理该token',
          '实现专家负载均衡'
        ]
      },
      'expert_computation': {
        title: '专家计算',
        content: '选中的专家对token进行处理。',
        points: [
          '每个专家是独立的FFN',
          '专家并行处理不同tokens',
          '提高模型容量和效率'
        ]
      },
    };

    return explanations[stepType] || {
      title: stepType,
      content: '暂无详细说明',
      points: []
    };
  };

  const explanation = currentStepData ? getDetailedExplanation(currentStepData.step_type) : null;

  return (
    <Card className="sticky top-24">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">详细解释</h3>
        </div>
        <button className="text-xs text-purple-400 hover:text-purple-300 transition-colors">
          中文 / EN
        </button>
      </div>

      {explanation && (
        <>
          {/* Current Step Explanation */}
          <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 mb-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                <span className="text-white font-bold text-sm">{currentStep + 1}</span>
              </div>
              <div>
                <h4 className="text-white font-medium mb-2">{explanation.title}</h4>
                <p className="text-sm text-slate-300 leading-relaxed mb-3">
                  {explanation.content}
                </p>
                {explanation.points.length > 0 && (
                  <ul className="space-y-1.5">
                    {explanation.points.map((point, idx) => (
                      <li key={idx} className="text-sm text-slate-400 flex items-start gap-2">
                        <span className="text-purple-400 mt-1">•</span>
                        <span>{point}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>

          {/* Key Concepts */}
          <div className="bg-slate-900/30 border border-slate-700/50 rounded-lg p-4 mb-4">
            <h4 className="text-sm font-medium text-slate-300 mb-3">关键概念</h4>
            <div className="space-y-2">
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-purple-400 rounded-full mt-1.5" />
                <div>
                  <span className="text-sm text-white font-medium">矩阵维度:</span>
                  <span className="text-sm text-slate-400 ml-2">
                    注意各个矩阵的形状变化
                  </span>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <div className="w-1.5 h-1.5 bg-pink-400 rounded-full mt-1.5" />
                <div>
                  <span className="text-sm text-white font-medium">计算复杂度:</span>
                  <span className="text-sm text-slate-400 ml-2">
                    O(n²d) 其中n是序列长度
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Related Resources */}
          <div>
            <h4 className="text-sm font-medium text-slate-400 mb-3">相关资源</h4>
            <div className="space-y-2">
              <a
                href="#"
                className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300 group transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                <span className="group-hover:underline">Attention Is All You Need 论文</span>
              </a>
              <a
                href="#"
                className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300 group transition-colors"
              >
                <Video className="w-4 h-4" />
                <span className="group-hover:underline">注意力机制视频讲解</span>
              </a>
              <a
                href="#"
                className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300 group transition-colors"
              >
                <BookOpen className="w-4 h-4" />
                <span className="group-hover:underline">深入理解Transformer</span>
              </a>
            </div>
          </div>
        </>
      )}
    </Card>
  );
}
