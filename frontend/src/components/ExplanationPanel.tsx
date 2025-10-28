'use client';

import { useVisualizationStore } from '@/store/visualizationStore';

export default function ExplanationPanel() {
  const { isInitialized, currentStepData, currentStep } = useVisualizationStore();

  if (!isInitialized) {
    return (
      <div className="bg-gray-100 rounded-lg shadow-md p-6 min-h-[400px]">
        <h3 className="text-xl font-bold text-gray-800 mb-4">解释面板</h3>
        <p className="text-gray-500">等待计算初始化...</p>
      </div>
    );
  }

  const getDetailedExplanation = (stepType: string): string => {
    const explanations: Record<string, string> = {
      'layer_norm_1': `
        <strong>Layer Normalization (注意力前)</strong>
        <p>在多头注意力机制之前，对输入进行Layer Normalization。</p>
        <ul>
          <li>目的: 稳定训练，加速收敛</li>
          <li>操作: 对每个样本的特征维度进行归一化</li>
          <li>公式: LN(x) = γ * (x - μ) / σ + β</li>
        </ul>
      `,
      'q_projection': `
        <strong>Query 投影</strong>
        <p>将输入通过线性变换生成Query矩阵。</p>
        <ul>
          <li>操作: Q = X @ W_q</li>
          <li>作用: Query用于表示"查询"信息</li>
          <li>维度: [序列长度, n_embd]</li>
        </ul>
      `,
      'k_projection': `
        <strong>Key 投影</strong>
        <p>将输入通过线性变换生成Key矩阵。</p>
        <ul>
          <li>操作: K = X @ W_k</li>
          <li>作用: Key用于被Query查询匹配</li>
          <li>维度: [序列长度, n_embd]</li>
        </ul>
      `,
      'v_projection': `
        <strong>Value 投影</strong>
        <p>将输入通过线性变换生成Value矩阵。</p>
        <ul>
          <li>操作: V = X @ W_v</li>
          <li>作用: Value存储实际要传递的信息</li>
          <li>维度: [序列长度, n_embd]</li>
        </ul>
      `,
      'attention_scores': `
        <strong>注意力分数计算</strong>
        <p>计算Query和Key之间的相似度。</p>
        <ul>
          <li>操作: scores = (Q @ K^T) / sqrt(d_k)</li>
          <li>作用: 衡量不同位置之间的相关性</li>
          <li>缩放: 除以sqrt(d_k)防止梯度消失</li>
        </ul>
      `,
      'attention_weights': `
        <strong>注意力权重 (Softmax)</strong>
        <p>对注意力分数应用Softmax，得到归一化的权重。</p>
        <ul>
          <li>操作: weights = softmax(scores)</li>
          <li>作用: 将分数转换为概率分布</li>
          <li>特性: 所有权重和为1，范围[0, 1]</li>
        </ul>
      `,
      'attention_output': `
        <strong>注意力输出</strong>
        <p>使用注意力权重对Value进行加权求和。</p>
        <ul>
          <li>操作: output = weights @ V</li>
          <li>作用: 聚合相关信息</li>
          <li>结果: 每个位置获得上下文信息</li>
        </ul>
      `,
      'attention_projection': `
        <strong>多头注意力输出投影</strong>
        <p>将多头注意力的输出进行线性变换。</p>
        <ul>
          <li>操作: output = attention_out @ W_o</li>
          <li>作用: 融合多个注意力头的信息</li>
          <li>维度: [序列长度, n_embd]</li>
        </ul>
      `,
      'residual_1': `
        <strong>残差连接 (Attention)</strong>
        <p>将注意力输出与原始输入相加。</p>
        <ul>
          <li>操作: x = x + attention_output</li>
          <li>作用: 保留原始信息，缓解梯度消失</li>
          <li>特性: 让模型学习残差而非直接映射</li>
        </ul>
      `,
      'layer_norm_2': `
        <strong>Layer Normalization (FFN前)</strong>
        <p>在前馈网络之前，对残差连接结果进行Layer Normalization。</p>
        <ul>
          <li>目的: 稳定训练</li>
          <li>位置: 残差连接后，FFN前</li>
        </ul>
      `,
      'ffn_hidden': `
        <strong>前馈网络 - 第一层</strong>
        <p>前馈网络的第一层线性变换。</p>
        <ul>
          <li>操作: hidden = x @ W1 + b1</li>
          <li>特点: 通常扩展维度（如4倍）</li>
          <li>作用: 提供非线性变换能力</li>
        </ul>
      `,
      'ffn_relu': `
        <strong>ReLU 激活函数</strong>
        <p>对前馈网络第一层的输出应用ReLU激活。</p>
        <ul>
          <li>操作: ReLU(x) = max(0, x)</li>
          <li>作用: 引入非线性</li>
          <li>特性: 简单高效，缓解梯度消失</li>
        </ul>
      `,
      'ffn_output': `
        <strong>前馈网络 - 第二层</strong>
        <p>前馈网络的第二层线性变换。</p>
        <ul>
          <li>操作: output = hidden @ W2 + b2</li>
          <li>特点: 降维回原始维度</li>
          <li>作用: 完成信息转换</li>
        </ul>
      `,
      'residual_2': `
        <strong>残差连接 (FFN)</strong>
        <p>将前馈网络输出与其输入相加。</p>
        <ul>
          <li>操作: x = x + ffn_output</li>
          <li>作用: 保留信息，促进梯度流动</li>
          <li>结果: 完成一个完整的Transformer层</li>
        </ul>
      `,
    };

    return explanations[stepType] || `<p>步骤: ${stepType}</p>`;
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 min-h-[400px] overflow-y-auto">
      <h3 className="text-xl font-bold text-gray-800 mb-4">解释面板</h3>

      {currentStepData && (
        <div className="space-y-4">
          {/* Current Step Summary */}
          <div className="bg-blue-50 border-l-4 border-blue-600 p-4 rounded">
            <h4 className="font-semibold text-blue-900 mb-2">
              步骤 {currentStep + 1}: {currentStepData.step_type}
            </h4>
            <p className="text-blue-800 text-sm">
              {currentStepData.description}
            </p>
          </div>

          {/* Detailed Explanation */}
          <div className="prose prose-sm max-w-none">
            <div
              className="text-gray-700 [&_strong]:text-gray-900 [&_strong]:text-lg [&_strong]:block [&_strong]:mb-2 [&_ul]:list-disc [&_ul]:ml-6 [&_ul]:mt-2 [&_li]:mb-1 [&_p]:mb-2"
              dangerouslySetInnerHTML={{
                __html: getDetailedExplanation(currentStepData.step_type),
              }}
            />
          </div>

          {/* Data Shape Information */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">数据形状:</h4>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">输入形状:</span>
                <span className="font-mono text-gray-800">
                  {currentStepData.input_data && Array.isArray(currentStepData.input_data)
                    ? `[${currentStepData.input_data.length}, ${
                        currentStepData.input_data[0]?.length || 0
                      }]`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">输出形状:</span>
                <span className="font-mono text-gray-800">
                  {currentStepData.output_data && Array.isArray(currentStepData.output_data)
                    ? `[${currentStepData.output_data.length}, ${
                        currentStepData.output_data[0]?.length || 0
                      }]`
                    : 'N/A'}
                </span>
              </div>
            </div>
          </div>

          {/* Layer Information */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-green-900 mb-2">层信息:</h4>
            <p className="text-sm text-green-800">
              当前位于第 <strong>{currentStepData.layer_index + 1}</strong> 层
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
