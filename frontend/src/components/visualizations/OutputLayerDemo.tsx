'use client';

import React, { useState, useMemo } from 'react';
import { OutputLayerViz } from './OutputLayerViz';

export const OutputLayerDemo: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);

  const sampleVocabulary = useMemo(() => [
    'hello', 'world', 'the', 'a', 'is', 'are', 'was', 'were', 'in', 'on',
    'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among',
    'this', 'that', 'these', 'those', 'I', 'you', 'he', 'she', 'it', 'we',
    'they', 'what', 'which', 'who', 'where', 'when', 'why', 'how', 'can', 'could',
    'will', 'would', 'should', 'may', 'might', 'must', 'have', 'has', 'had',
  ], []);

  const sampleHiddenState = useMemo(() => 
    Array.from({ length: 5 }, () =>
      Array.from({ length: 768 }, () => (Math.random() - 0.5) * 2)
    ), 
  []);

  const handleStart = () => {
    setIsRunning(true);
  };

  const handleComplete = () => {
    console.log('Output layer visualization completed!');
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg shadow-xl p-8 text-white">
        <h1 className="text-4xl font-bold mb-4">输出层可视化演示</h1>
        <p className="text-lg mb-6">
          展示Transformer模型的最终输出层：Layer Normalization、Token选择、Logits Head、Softmax归一化和预测结果
        </p>
        
        {!isRunning && (
          <button
            onClick={handleStart}
            className="bg-white text-purple-600 font-bold py-3 px-8 rounded-lg hover:bg-gray-100 transition-all transform hover:scale-105 shadow-lg"
          >
            开始演示
          </button>
        )}
      </div>

      {isRunning && (
        <div className="space-y-6">
          <div className="bg-blue-50 border-l-4 border-blue-600 p-6 rounded-lg">
            <h3 className="text-lg font-bold text-blue-900 mb-2">输出层处理流程</h3>
            <ol className="list-decimal list-inside space-y-2 text-blue-800">
              <li><strong>Final Layer Normalization</strong>: 对最后一层的输出进行归一化</li>
              <li><strong>Token Selection</strong>: 选择最后一个token用于预测（Next Token Prediction）</li>
              <li><strong>Logits Head</strong>: 将选中的向量投影到词汇表空间</li>
              <li><strong>Softmax</strong>: 将logits转换为概率分布</li>
              <li><strong>Prediction</strong>: 选择概率最高的token作为预测结果</li>
            </ol>
          </div>

          <OutputLayerViz
            finalHiddenState={sampleHiddenState}
            vocabulary={sampleVocabulary}
            onComplete={handleComplete}
          />

          <div className="bg-green-50 border-l-4 border-green-600 p-6 rounded-lg">
            <h3 className="text-lg font-bold text-green-900 mb-2">技术说明</h3>
            <div className="space-y-2 text-green-800">
              <p>
                <strong>Next Token Prediction</strong>: 
                对于语言建模任务，我们使用序列中最后一个token的输出向量来预测下一个token。
              </p>
              <p>
                <strong>Logits Head</strong>: 
                一个线性层，将模型的隐藏维度（如768）投影到词汇表大小（如50257）。
                每个维度对应词汇表中一个token的未归一化分数。
              </p>
              <p>
                <strong>Softmax归一化</strong>: 
                将所有logits通过exp()函数和归一化，转换为概率分布，所有概率之和为1。
                公式：P(token_i) = exp(logit_i) / Σ exp(logit_j)
              </p>
              <p>
                <strong>Top-K显示</strong>: 
                由于词汇表通常非常大（数万个token），我们只显示概率最高的前K个候选token。
              </p>
            </div>
          </div>

          <div className="bg-purple-50 border-l-4 border-purple-600 p-6 rounded-lg">
            <h3 className="text-lg font-bold text-purple-900 mb-2">交互功能</h3>
            <ul className="list-disc list-inside space-y-2 text-purple-800">
              <li><strong>悬停显示</strong>: 将鼠标悬停在柱状图上，查看详细的token、logit和概率信息</li>
              <li><strong>搜索功能</strong>: 使用搜索框查找特定token的概率（在logits和预测阶段可用）</li>
              <li><strong>Top-K列表</strong>: 查看概率最高的10个候选token及其概率分布</li>
              <li><strong>高亮显示</strong>: 最高概率的token会以金色高亮显示，并带有特殊效果</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};
