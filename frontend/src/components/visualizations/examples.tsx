/**
 * 示例文件 - 展示如何使用各个可视化组件
 * 
 * 这个文件包含了所有组件的使用示例，可以作为参考或直接复制使用
 */

'use client';

import { useState, useMemo } from 'react';
import { TokenizationViz } from './TokenizationViz';
import { EmbeddingViz } from './EmbeddingViz';
import { TokenEmbeddingVisualization } from './TokenEmbeddingVisualization';
import { MultiHeadAttentionViz } from './MultiHeadAttentionViz';

// ============================================================================
// 示例 1: 基础词元化可视化
// ============================================================================

export const Example1_BasicTokenization = () => {
  const text = 'Hello World';
  const tokens = [15496, 2159];
  const tokenTexts = ['Hello', 'World'];

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 1: 基础词元化</h2>
      <TokenizationViz
        text={text}
        tokens={tokens}
        tokenTexts={tokenTexts}
        onComplete={() => console.log('词元化完成')}
      />
    </div>
  );
};

// ============================================================================
// 示例 2: 多词元可视化
// ============================================================================

export const Example2_MultipleTokens = () => {
  const text = 'The quick brown fox';
  const tokens = [464, 2068, 7586, 21831];
  const tokenTexts = ['The', 'quick', 'brown', 'fox'];

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 2: 多词元</h2>
      <TokenizationViz
        text={text}
        tokens={tokens}
        tokenTexts={tokenTexts}
        onComplete={() => console.log('多词元处理完成')}
      />
    </div>
  );
};

// ============================================================================
// 示例 3: 嵌入可视化 - 串行模式
// ============================================================================

export const Example3_EmbeddingSerial = () => {
  const tokens = [15496, 2159];
  const tokenTexts = ['Hello', 'World'];
  const nEmbd = 768;
  const nVocab = 50257;

  // 生成随机嵌入向量
  const embeddings = tokens.map(() => 
    Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
  );

  // 生成位置编码
  const positionalEncodings = tokens.map((_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 3: 嵌入可视化 - 串行模式</h2>
      <EmbeddingViz
        tokens={tokens}
        tokenTexts={tokenTexts}
        embeddings={embeddings}
        positionalEncodings={positionalEncodings}
        nEmbd={nEmbd}
        nVocab={nVocab}
        animationMode="serial"
        onComplete={() => console.log('串行嵌入完成')}
      />
    </div>
  );
};

// ============================================================================
// 示例 4: 嵌入可视化 - 并行模式
// ============================================================================

export const Example4_EmbeddingParallel = () => {
  const tokens = [15496, 2159];
  const tokenTexts = ['Hello', 'World'];
  const nEmbd = 768;
  const nVocab = 50257;

  const embeddings = tokens.map(() => 
    Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
  );

  const positionalEncodings = tokens.map((_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 4: 嵌入可视化 - 并行模式</h2>
      <EmbeddingViz
        tokens={tokens}
        tokenTexts={tokenTexts}
        embeddings={embeddings}
        positionalEncodings={positionalEncodings}
        nEmbd={nEmbd}
        nVocab={nVocab}
        animationMode="parallel"
        onComplete={() => console.log('并行嵌入完成')}
      />
    </div>
  );
};

// ============================================================================
// 示例 5: 完整流程可视化
// ============================================================================

export const Example5_FullPipeline = () => {
  const text = 'Transformer model';
  const tokens = [8291, 16354, 2746];
  const tokenTexts = ['Trans', 'former', 'model'];
  const nEmbd = 768;
  const nVocab = 50257;

  const embeddings = tokens.map(() => 
    Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
  );

  const positionalEncodings = tokens.map((_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 5: 完整流程</h2>
      <TokenEmbeddingVisualization
        text={text}
        tokens={tokens}
        tokenTexts={tokenTexts}
        embeddings={embeddings}
        positionalEncodings={positionalEncodings}
        nEmbd={nEmbd}
        nVocab={nVocab}
      />
    </div>
  );
};

// ============================================================================
// 示例 6: 小嵌入维度 (用于测试)
// ============================================================================

export const Example6_SmallDimension = () => {
  const tokens = [100, 200];
  const tokenTexts = ['A', 'B'];
  const nEmbd = 32;  // 小维度，方便查看
  const nVocab = 1000;

  const embeddings = tokens.map(() => 
    Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
  );

  const positionalEncodings = tokens.map((_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 6: 小维度 (32)</h2>
      <TokenEmbeddingVisualization
        text="A B"
        tokens={tokens}
        tokenTexts={tokenTexts}
        embeddings={embeddings}
        positionalEncodings={positionalEncodings}
        nEmbd={nEmbd}
        nVocab={nVocab}
      />
    </div>
  );
};

// ============================================================================
// 示例 7: 中文文本
// ============================================================================

export const Example7_ChineseText = () => {
  const text = '你好世界';
  const tokens = [19526, 25001];
  const tokenTexts = ['你好', '世界'];
  const nEmbd = 768;
  const nVocab = 50257;

  const embeddings = tokens.map(() => 
    Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
  );

  const positionalEncodings = tokens.map((_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 7: 中文文本</h2>
      <TokenEmbeddingVisualization
        text={text}
        tokens={tokens}
        tokenTexts={tokenTexts}
        embeddings={embeddings}
        positionalEncodings={positionalEncodings}
        nEmbd={nEmbd}
        nVocab={nVocab}
      />
    </div>
  );
};

// ============================================================================
// 示例 8: 长文本 (5+ 词元)
// ============================================================================

export const Example8_LongText = () => {
  const text = 'The quick brown fox jumps over';
  const tokens = [464, 2068, 7586, 21831, 18045, 625];
  const tokenTexts = ['The', 'quick', 'brown', 'fox', 'jumps', 'over'];
  const nEmbd = 768;
  const nVocab = 50257;

  const embeddings = tokens.map(() => 
    Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
  );

  const positionalEncodings = tokens.map((_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 8: 长文本 (6 词元)</h2>
      <TokenEmbeddingVisualization
        text={text}
        tokens={tokens}
        tokenTexts={tokenTexts}
        embeddings={embeddings}
        positionalEncodings={positionalEncodings}
        nEmbd={nEmbd}
        nVocab={nVocab}
      />
    </div>
  );
};

// ============================================================================
// 示例 9: 多头自注意力机制
// ============================================================================

export const Example9_MultiHeadAttention = () => {
  const nTokens = 4;
  const nEmbd = 64;
  const nHead = 4;
  const dK = nEmbd / nHead;

  // 使用 useMemo 确保数据只生成一次
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

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">示例 9: 多头自注意力机制</h2>
      <p className="text-sm text-gray-600 mb-4">
        展示完整的多头自注意力计算过程，包括 Layer Norm、Q/K/V 生成、注意力计算和残差连接
      </p>
      <MultiHeadAttentionViz
        inputData={inputData}
        weights={weights}
        config={config}
        tokenTexts={tokenTexts}
        animationMode="serial"
        onComplete={() => console.log('多头注意力可视化完成')}
      />
    </div>
  );
};

// ============================================================================
// 示例合集组件
// ============================================================================

export const AllExamples = () => {
  const [currentExample, setCurrentExample] = useState(1);

  const examples = [
    { id: 1, name: '基础词元化', component: Example1_BasicTokenization },
    { id: 2, name: '多词元', component: Example2_MultipleTokens },
    { id: 3, name: '嵌入-串行', component: Example3_EmbeddingSerial },
    { id: 4, name: '嵌入-并行', component: Example4_EmbeddingParallel },
    { id: 5, name: '完整流程', component: Example5_FullPipeline },
    { id: 6, name: '小维度', component: Example6_SmallDimension },
    { id: 7, name: '中文文本', component: Example7_ChineseText },
    { id: 8, name: '长文本', component: Example8_LongText },
    { id: 9, name: '多头注意力', component: Example9_MultiHeadAttention },
  ];

  const CurrentComponent = examples.find(ex => ex.id === currentExample)?.component || Example1_BasicTokenization;

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h1 className="text-2xl font-bold mb-4">D3.js 可视化示例合集</h1>
          <div className="flex flex-wrap gap-2">
            {examples.map(ex => (
              <button
                key={ex.id}
                onClick={() => setCurrentExample(ex.id)}
                className={`px-4 py-2 rounded ${
                  currentExample === ex.id
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {ex.name}
              </button>
            ))}
          </div>
        </div>
        <CurrentComponent />
      </div>
    </div>
  );
};

// ============================================================================
// 辅助函数：生成测试数据
// ============================================================================

export const generateTestData = (
  numTokens: number = 3,
  nEmbd: number = 768,
  nVocab: number = 50257
) => {
  const text = Array.from({ length: numTokens }, (_, i) => `token${i}`).join(' ');
  const tokens = Array.from({ length: numTokens }, (_, i) => 1000 + i);
  const tokenTexts = Array.from({ length: numTokens }, (_, i) => `token${i}`);

  const embeddings = tokens.map(() => 
    Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
  );

  const positionalEncodings = tokens.map((_, pos) => 
    Array.from({ length: nEmbd }, (_, i) => {
      const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
      return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );

  return {
    text,
    tokens,
    tokenTexts,
    embeddings,
    positionalEncodings,
    nEmbd,
    nVocab,
  };
};
