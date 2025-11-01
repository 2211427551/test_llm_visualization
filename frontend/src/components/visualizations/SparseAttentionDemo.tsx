'use client';

import { useState, useMemo } from 'react';
import { SparseAttentionViz } from './SparseAttentionViz';

export const SparseAttentionDemo: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [key, setKey] = useState(0);
  const [sparsePattern, setSparsePattern] = useState<string>('sliding_window');
  const [windowSize, setWindowSize] = useState<number>(3);
  const [blockSize, setBlockSize] = useState<number>(4);
  const [showComparison, setShowComparison] = useState<boolean>(false);

  // Generate test data
  const nTokens = 8;
  const nEmbd = 64;
  const nHead = 4;
  const dK = nEmbd / nHead;

  const inputData = useMemo(() => {
    const generateRandomMatrix = (rows: number, cols: number): number[][] => {
      return Array(rows).fill(0).map(() => 
        Array(cols).fill(0).map(() => (Math.random() - 0.5) * 0.5)
      );
    };
    return generateRandomMatrix(nTokens, nEmbd);
  }, [nTokens, nEmbd]);

  const weights = useMemo(() => {
    const generateRandomMatrix = (rows: number, cols: number): number[][] => {
      return Array(rows).fill(0).map(() => 
        Array(cols).fill(0).map(() => (Math.random() - 0.5) * 0.5)
      );
    };
    const ln_gamma = Array(nEmbd).fill(0).map(() => 1 + Math.random() * 0.1);
    const ln_beta = Array(nEmbd).fill(0).map(() => Math.random() * 0.1);
    return {
      wq: generateRandomMatrix(nEmbd, nEmbd),
      wk: generateRandomMatrix(nEmbd, nEmbd),
      wv: generateRandomMatrix(nEmbd, nEmbd),
      wo: generateRandomMatrix(nEmbd, nEmbd),
      ln_gamma,
      ln_beta,
    };
  }, [nEmbd]);

  const config = {
    n_head: nHead,
    d_k: dK,
    sparse_pattern: sparsePattern,
    window_size: windowSize,
    block_size: blockSize,
    global_tokens: sparsePattern === 'global_local' ? [0] : undefined,
  };

  const handleReplay = () => {
    setIsPlaying(false);
    setTimeout(() => {
      setKey(prev => prev + 1);
      setIsPlaying(true);
    }, 100);
  };

  const handlePatternChange = (pattern: string) => {
    setSparsePattern(pattern);
    setKey(prev => prev + 1);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            稀疏注意力可视化 (Sparse Attention)
          </h1>
          <p className="text-lg text-gray-700 mb-4">
            展示现代LLM如何通过稀疏化提高效率
          </p>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h2 className="text-lg font-semibold text-blue-900 mb-2">关键概念</h2>
            <ul className="list-disc list-inside text-blue-800 space-y-1">
              <li><strong>密集注意力</strong>: 每个token关注所有token，复杂度O(n²)</li>
              <li><strong>稀疏注意力</strong>: 每个token只关注部分token，大幅降低计算量</li>
              <li><strong>滑动窗口</strong>: 只关注附近的token，适合局部依赖</li>
              <li><strong>全局+局部</strong>: 特殊token全局关注，普通token局部关注</li>
              <li><strong>分块</strong>: 按块组织关注模式，平衡效率和效果</li>
            </ul>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">控制面板</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                稀疏模式
              </label>
              <select
                value={sparsePattern}
                onChange={(e) => handlePatternChange(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="dense">Dense (标准注意力)</option>
                <option value="sliding_window">Sliding Window (滑动窗口)</option>
                <option value="global_local">Global + Local (全局+局部)</option>
                <option value="blocked">Blocked (分块)</option>
              </select>
            </div>

            {(sparsePattern === 'sliding_window' || sparsePattern === 'global_local') && (
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  窗口大小: {windowSize}
                </label>
                <input
                  type="range"
                  min="1"
                  max="5"
                  value={windowSize}
                  onChange={(e) => {
                    setWindowSize(parseInt(e.target.value));
                    setKey(prev => prev + 1);
                  }}
                  className="w-full"
                />
              </div>
            )}

            {sparsePattern === 'blocked' && (
              <div>
                <label className="block text-sm font-semibold text-gray-700 mb-2">
                  块大小: {blockSize}
                </label>
                <input
                  type="range"
                  min="2"
                  max="6"
                  value={blockSize}
                  onChange={(e) => {
                    setBlockSize(parseInt(e.target.value));
                    setKey(prev => prev + 1);
                  }}
                  className="w-full"
                />
              </div>
            )}

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                显示对比
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={showComparison}
                  onChange={(e) => setShowComparison(e.target.checked)}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <span className="text-sm text-gray-700">密集 vs 稀疏对比</span>
              </label>
            </div>
          </div>

          <div className="mt-6 flex gap-4">
            <button
              onClick={handleReplay}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-md"
            >
              {isPlaying ? '重播动画' : '开始动画'}
            </button>
          </div>

          <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600">Token数量</div>
              <div className="text-2xl font-bold text-gray-900">{nTokens}</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600">嵌入维度</div>
              <div className="text-2xl font-bold text-gray-900">{nEmbd}</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600">注意力头数</div>
              <div className="text-2xl font-bold text-gray-900">{nHead}</div>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="text-sm text-gray-600">每头维度</div>
              <div className="text-2xl font-bold text-gray-900">{dK}</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">可视化</h2>
          {isPlaying && (
            <SparseAttentionViz
              key={key}
              inputData={inputData}
              weights={weights}
              config={config}
              showComparison={showComparison}
              onComplete={() => console.log('Animation complete')}
            />
          )}
          {!isPlaying && (
            <div className="text-center py-12 text-gray-500">
              点击"开始动画"按钮查看稀疏注意力的可视化过程
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">稀疏模式说明</h2>
          
          <div className="space-y-4">
            <div className="border-l-4 border-blue-500 pl-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                1. Sliding Window (滑动窗口)
              </h3>
              <p className="text-gray-700 mb-2">
                每个token只关注前后w个token。适用于局部依赖强的任务，如文本生成。
              </p>
              <p className="text-sm text-gray-600">
                复杂度: O(n·w) | 应用: Longformer, BigBird
              </p>
            </div>

            <div className="border-l-4 border-green-500 pl-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                2. Global + Local (全局+局部)
              </h3>
              <p className="text-gray-700 mb-2">
                特殊token(如[CLS])可以关注所有token，普通token只关注局部窗口。
              </p>
              <p className="text-sm text-gray-600">
                复杂度: O(n·w + g·n) | 应用: Longformer, LED
              </p>
            </div>

            <div className="border-l-4 border-purple-500 pl-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                3. Blocked (分块)
              </h3>
              <p className="text-gray-700 mb-2">
                将序列分成块，token只关注同块和相邻块的token。
              </p>
              <p className="text-sm text-gray-600">
                复杂度: O(n·b) | 应用: BlockBERT, Blockwise Attention
              </p>
            </div>

            <div className="border-l-4 border-gray-500 pl-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                4. Dense (标准注意力)
              </h3>
              <p className="text-gray-700 mb-2">
                每个token关注所有token，无稀疏化。
              </p>
              <p className="text-sm text-gray-600">
                复杂度: O(n²) | 应用: BERT, GPT-2, 标准Transformer
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <h2 className="text-xl font-bold text-yellow-900 mb-3">技术实现说明</h2>
          <ul className="list-disc list-inside text-yellow-800 space-y-2">
            <li>使用D3.js v7实现动画和可视化</li>
            <li>TypeScript确保类型安全</li>
            <li>支持多种稀疏模式切换</li>
            <li>热力图展示注意力权重</li>
            <li>掩码覆盖层展示稀疏模式</li>
            <li>交互式悬停查看数值</li>
            <li>实时计算稀疏度指标</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
