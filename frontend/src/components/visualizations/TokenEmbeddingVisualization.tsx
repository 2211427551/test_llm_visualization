'use client';

import { useState } from 'react';
import { TokenizationViz } from './TokenizationViz';
import { EmbeddingViz } from './EmbeddingViz';

interface TokenEmbeddingVisualizationProps {
  text: string;
  tokens: number[];
  tokenTexts: string[];
  embeddings: number[][];
  positionalEncodings: number[][];
  nEmbd: number;
  nVocab: number;
}

export const TokenEmbeddingVisualization: React.FC<TokenEmbeddingVisualizationProps> = ({
  text,
  tokens,
  tokenTexts,
  embeddings,
  positionalEncodings,
  nEmbd,
  nVocab,
}) => {
  const [stage, setStage] = useState<'tokenization' | 'embedding' | 'complete'>('tokenization');
  const [animationMode, setAnimationMode] = useState<'serial' | 'parallel'>('serial');
  const showControls = true;

  const handleTokenizationComplete = () => {
    setTimeout(() => {
      setStage('embedding');
    }, 1000);
  };

  const handleEmbeddingComplete = () => {
    setStage('complete');
  };

  const reset = () => {
    setStage('tokenization');
  };

  return (
    <div className="w-full space-y-4">
      {showControls && (
        <div className="flex items-center justify-between bg-white rounded-lg shadow p-4">
          <div className="flex items-center space-x-4">
            <div className="text-sm">
              <span className="font-semibold text-gray-700">当前阶段: </span>
              <span className="text-blue-600">
                {stage === 'tokenization' && '词元化'}
                {stage === 'embedding' && '嵌入与位置编码'}
                {stage === 'complete' && '完成'}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-sm font-semibold text-gray-700">动画模式:</span>
              <button
                onClick={() => setAnimationMode('serial')}
                className={`px-3 py-1 text-xs rounded ${
                  animationMode === 'serial'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                disabled={stage !== 'tokenization'}
              >
                串行
              </button>
              <button
                onClick={() => setAnimationMode('parallel')}
                className={`px-3 py-1 text-xs rounded ${
                  animationMode === 'parallel'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                disabled={stage !== 'tokenization'}
              >
                并行
              </button>
            </div>
          </div>
          <button
            onClick={reset}
            className="px-4 py-2 bg-green-500 text-white text-sm rounded hover:bg-green-600 transition-colors"
          >
            🔄 重新播放
          </button>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-lg p-6">
        {stage === 'tokenization' && (
          <div>
            <h3 className="text-lg font-bold text-gray-800 mb-4">
              步骤 1: 词元化（Tokenization）
            </h3>
            <TokenizationViz
              text={text}
              tokens={tokens}
              tokenTexts={tokenTexts}
              onComplete={handleTokenizationComplete}
            />
          </div>
        )}

        {stage === 'embedding' && (
          <div>
            <h3 className="text-lg font-bold text-gray-800 mb-4">
              步骤 2: 嵌入查表与位置编码（Embedding + Positional Encoding）
            </h3>
            <EmbeddingViz
              tokens={tokens}
              tokenTexts={tokenTexts}
              embeddings={embeddings}
              positionalEncodings={positionalEncodings}
              nEmbd={nEmbd}
              nVocab={nVocab}
              animationMode={animationMode}
              onComplete={handleEmbeddingComplete}
            />
          </div>
        )}

        {stage === 'complete' && (
          <div className="text-center py-12">
            <div className="inline-block p-4 bg-green-100 rounded-full mb-4">
              <svg
                className="w-16 h-16 text-green-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-gray-800 mb-2">
              词元化与嵌入完成！
            </h3>
            <p className="text-gray-600 mb-6">
              输入向量已准备好进入 Transformer 层
            </p>
            <div className="bg-gray-50 rounded-lg p-4 inline-block">
              <div className="text-sm text-gray-700">
                <div className="mb-2">
                  <span className="font-semibold">词元数量:</span> {tokens.length}
                </div>
                <div className="mb-2">
                  <span className="font-semibold">嵌入维度:</span> {nEmbd}
                </div>
                <div>
                  <span className="font-semibold">输出形状:</span> [{tokens.length}, {nEmbd}]
                </div>
              </div>
            </div>
            <div className="mt-6">
              <button
                onClick={reset}
                className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
              >
                重新播放动画
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
