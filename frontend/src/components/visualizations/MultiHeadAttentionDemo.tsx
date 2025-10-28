'use client';

import { useState, useMemo } from 'react';
import { MultiHeadAttentionViz } from './MultiHeadAttentionViz';

export const MultiHeadAttentionDemo: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [animationMode, setAnimationMode] = useState<'serial' | 'parallel'>('serial');

  // 配置参数
  const nTokens = 4;
  const nEmbd = 64;
  const nHead = 4;
  const dK = nEmbd / nHead;

  // 生成测试数据 - 使用 useMemo 确保不会在每次渲染时重新生成
  const { inputData, weights, config, tokenTexts } = useMemo(() => {
    // 生成随机矩阵
    const generateRandomMatrix = (rows: number, cols: number, scale = 1): number[][] => {
      return Array.from({ length: rows }, () =>
        Array.from({ length: cols }, () => (Math.random() - 0.5) * 2 * scale)
      );
    };

    // 生成随机向量
    const generateRandomVector = (length: number, mean = 0, scale = 1): number[] => {
      return Array.from({ length }, () => mean + (Math.random() - 0.5) * scale);
    };

    return {
      inputData: generateRandomMatrix(nTokens, nEmbd, 0.5),
      weights: {
        wq: generateRandomMatrix(nEmbd, nEmbd, 0.1),
        wk: generateRandomMatrix(nEmbd, nEmbd, 0.1),
        wv: generateRandomMatrix(nEmbd, nEmbd, 0.1),
        wo: generateRandomMatrix(nEmbd, nEmbd, 0.1),
        ln_gamma: generateRandomVector(nEmbd, 1.0, 0.1),
        ln_beta: generateRandomVector(nEmbd, 0, 0.1),
      },
      config: {
        n_head: nHead,
        d_k: dK,
      },
      tokenTexts: ['The', 'cat', 'sat', 'down'],
    };
  }, [nTokens, nEmbd, nHead, dK]);

  const handleComplete = () => {
    setIsPlaying(false);
  };

  const handleReplay = () => {
    setIsPlaying(false);
    setTimeout(() => setIsPlaying(true), 100);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Multi-Head Self-Attention 可视化
          </h1>
          <p className="text-gray-600 mb-6">
            完整展示标准多头自注意力机制的计算过程，包括 Layer Normalization、Q/K/V 生成、
            注意力计算、多头合并和残差连接。
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="font-semibold text-blue-900 mb-2">配置信息</h3>
              <div className="text-sm text-blue-800 space-y-1">
                <div>词元数 (n_tokens): {nTokens}</div>
                <div>嵌入维度 (n_embd): {nEmbd}</div>
                <div>注意力头数 (n_head): {nHead}</div>
                <div>每个头维度 (d_k): {dK}</div>
              </div>
            </div>

            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="font-semibold text-green-900 mb-2">可视化步骤</h3>
              <ol className="text-sm text-green-800 space-y-1 list-decimal list-inside">
                <li>Layer Normalization</li>
                <li>Q, K, V 矩阵生成</li>
                <li>分割成多个注意力头</li>
                <li>每个头的注意力计算</li>
                <li>多头输出拼接</li>
                <li>输出线性变换</li>
                <li>残差连接</li>
              </ol>
            </div>
          </div>

          <div className="flex items-center gap-4 mb-6">
            <button
              onClick={handleReplay}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-semibold shadow-md"
            >
              {isPlaying ? '重新播放' : '开始播放'}
            </button>

            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-gray-700">动画模式:</label>
              <select
                value={animationMode}
                onChange={(e) => setAnimationMode(e.target.value as 'serial' | 'parallel')}
                className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="serial">串行</option>
                <option value="parallel">并行</option>
              </select>
            </div>
          </div>

          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-yellow-700">
                  提示：由于计算复杂度，仅显示部分权重矩阵和前几个注意力头。
                  完整实现会处理所有头并进行实际的矩阵运算。
                </p>
              </div>
            </div>
          </div>
        </div>

        {isPlaying && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <MultiHeadAttentionViz
              inputData={inputData}
              weights={weights}
              config={config}
              tokenTexts={tokenTexts}
              animationMode={animationMode}
              onComplete={handleComplete}
            />
          </div>
        )}

        {!isPlaying && (
          <div className="bg-white rounded-lg shadow-lg p-12 text-center">
            <svg
              className="mx-auto h-24 w-24 text-gray-400 mb-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
              />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <h3 className="text-xl font-semibold text-gray-700 mb-2">
              准备开始
            </h3>
            <p className="text-gray-500">
              点击&quot;开始播放&quot;按钮查看多头自注意力机制的完整可视化过程
            </p>
          </div>
        )}

        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            关于多头自注意力机制
          </h2>
          
          <div className="prose prose-blue max-w-none">
            <h3 className="text-lg font-semibold text-gray-800 mt-4 mb-2">
              什么是多头自注意力？
            </h3>
            <p className="text-gray-600 mb-4">
              多头自注意力（Multi-Head Self-Attention）是 Transformer 模型的核心组件。
              它允许模型同时关注输入序列的不同位置和不同的表示子空间。
            </p>

            <h3 className="text-lg font-semibold text-gray-800 mt-4 mb-2">
              核心概念
            </h3>
            <ul className="text-gray-600 space-y-2 list-disc list-inside">
              <li>
                <strong>Query (Q), Key (K), Value (V)</strong>: 
                通过线性变换从输入生成，用于计算注意力权重和输出
              </li>
              <li>
                <strong>注意力分数</strong>: 
                计算 Q 和 K 的点积，衡量不同位置之间的相关性
              </li>
              <li>
                <strong>Softmax</strong>: 
                将注意力分数归一化为概率分布
              </li>
              <li>
                <strong>多头机制</strong>: 
                将嵌入维度分割成多个头，每个头学习不同的注意力模式
              </li>
              <li>
                <strong>残差连接</strong>: 
                将原始输入加到输出上，帮助梯度传播
              </li>
            </ul>

            <h3 className="text-lg font-semibold text-gray-800 mt-4 mb-2">
              数学公式
            </h3>
            <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm text-gray-800 space-y-2">
              <div>Attention(Q, K, V) = softmax(QK<sup>T</sup> / √d<sub>k</sub>) V</div>
              <div>MultiHead(Q, K, V) = Concat(head<sub>1</sub>, ..., head<sub>h</sub>) W<sup>O</sup></div>
              <div>where head<sub>i</sub> = Attention(QW<sub>i</sub><sup>Q</sup>, KW<sub>i</sub><sup>K</sup>, VW<sub>i</sub><sup>V</sup>)</div>
            </div>

            <h3 className="text-lg font-semibold text-gray-800 mt-4 mb-2">
              交互提示
            </h3>
            <ul className="text-gray-600 space-y-2 list-disc list-inside">
              <li>将鼠标悬停在矩阵单元格上查看具体数值</li>
              <li>注意力权重矩阵显示了词元之间的关注程度</li>
              <li>不同的颜色代表不同的注意力头</li>
              <li>观察每个头如何学习不同的注意力模式</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
