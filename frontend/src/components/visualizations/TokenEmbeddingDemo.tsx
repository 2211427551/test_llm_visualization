'use client';

import { useState } from 'react';
import { TokenEmbeddingVisualization } from './TokenEmbeddingVisualization';

export const TokenEmbeddingDemo = () => {
  const [inputText, setInputText] = useState('hello world');
  const [isStarted, setIsStarted] = useState(false);

  const demoTokenTexts = ['hello', 'world'];
  const demoTokens = [31373, 995];
  const nEmbd = 768;
  const nVocab = 50257;

  const generateMockEmbeddings = () => {
    return demoTokens.map(() => 
      Array.from({ length: nEmbd }, () => (Math.random() - 0.5) * 2)
    );
  };

  const generatePositionalEncodings = () => {
    return demoTokens.map((_, pos) => 
      Array.from({ length: nEmbd }, (_, i) => {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / nEmbd);
        return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      })
    );
  };

  const [embeddings] = useState(generateMockEmbeddings());
  const [positionalEncodings] = useState(generatePositionalEncodings());

  const handleStart = () => {
    setIsStarted(true);
  };

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg shadow-lg p-6 text-white">
        <h1 className="text-3xl font-bold mb-2">
          Transformer 词元化与嵌入可视化
        </h1>
        <p className="text-blue-100">
          使用 D3.js 展示 Tokenization 和 Embedding 过程，包括位置编码
        </p>
      </div>

      {!isStarted ? (
        <div className="bg-white rounded-lg shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">演示设置</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                输入文本:
              </label>
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="输入文本..."
              />
            </div>
            <div className="bg-blue-50 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-blue-900 mb-2">配置信息:</h3>
              <div className="grid grid-cols-2 gap-2 text-sm text-blue-800">
                <div>词汇表大小: {nVocab.toLocaleString()}</div>
                <div>嵌入维度: {nEmbd}</div>
                <div>词元数量: {demoTokenTexts.length}</div>
                <div>输出形状: [{demoTokenTexts.length}, {nEmbd}]</div>
              </div>
            </div>
            <button
              onClick={handleStart}
              className="w-full px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-semibold text-lg"
            >
              🚀 开始可视化
            </button>
          </div>
        </div>
      ) : (
        <TokenEmbeddingVisualization
          text={inputText}
          tokens={demoTokens}
          tokenTexts={demoTokenTexts}
          embeddings={embeddings}
          positionalEncodings={positionalEncodings}
          nEmbd={nEmbd}
          nVocab={nVocab}
        />
      )}

      <div className="bg-gray-50 rounded-lg shadow p-6">
        <h2 className="text-lg font-bold text-gray-800 mb-3">关于此可视化</h2>
        <div className="space-y-2 text-sm text-gray-700">
          <p>
            <strong>词元化 (Tokenization):</strong> 将输入文本分割成独立的词元，每个词元被映射到一个唯一的 ID。
          </p>
          <p>
            <strong>嵌入查表 (Embedding Lookup):</strong> 使用词元 ID 从嵌入矩阵中查找对应的向量表示。
          </p>
          <p>
            <strong>位置编码 (Positional Encoding):</strong> 为每个位置生成独特的编码向量，使用正弦/余弦函数。
          </p>
          <p>
            <strong>向量相加 (Addition):</strong> 将嵌入向量和位置编码按元素相加，得到最终的输入向量。
          </p>
        </div>
      </div>
    </div>
  );
};
