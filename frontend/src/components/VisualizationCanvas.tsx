'use client';

import { useState } from 'react';
import { useVisualizationStore } from '@/store/visualizationStore';
import { TokenEmbeddingVisualization } from './visualizations';

export default function VisualizationCanvas() {
  const { 
    isInitialized, 
    currentStepData, 
    tokenTexts, 
    tokens, 
    inputText,
    config 
  } = useVisualizationStore();
  
  const [showD3Viz, setShowD3Viz] = useState(true);
  const [embeddings, setEmbeddings] = useState<number[][]>([]);
  const [positionalEncodings, setPositionalEncodings] = useState<number[][]>([]);

  if (!isInitialized) {
    return (
      <div className="bg-gray-100 rounded-lg shadow-md p-8 flex items-center justify-center min-h-[400px]">
        <div className="text-center text-gray-500">
          <svg
            className="w-16 h-16 mx-auto mb-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <p className="text-lg font-medium">可视化画布</p>
          <p className="text-sm">等待数据初始化...</p>
        </div>
      </div>
    );
  }

  const generateMockEmbeddings = () => {
    if (embeddings.length > 0) return;
    
    const mockEmbeddings = tokens.map(() => 
      Array.from({ length: config.n_embd }, () => (Math.random() - 0.5) * 2)
    );
    
    const mockPositionalEncodings = tokens.map((_, pos) => 
      Array.from({ length: config.n_embd }, (_, i) => {
        const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / config.n_embd);
        return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
      })
    );
    
    setEmbeddings(mockEmbeddings);
    setPositionalEncodings(mockPositionalEncodings);
  };

  if (showD3Viz && embeddings.length === 0) {
    generateMockEmbeddings();
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 min-h-[400px]">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-bold text-gray-800">可视化区域</h3>
        <div className="flex space-x-2">
          <button
            onClick={() => setShowD3Viz(true)}
            className={`px-3 py-1 text-sm rounded ${
              showD3Viz
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            D3.js 动画
          </button>
          <button
            onClick={() => setShowD3Viz(false)}
            className={`px-3 py-1 text-sm rounded ${
              !showD3Viz
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            数据视图
          </button>
        </div>
      </div>

      {showD3Viz && embeddings.length > 0 ? (
        <TokenEmbeddingVisualization
          text={inputText}
          tokens={tokens}
          tokenTexts={tokenTexts}
          embeddings={embeddings}
          positionalEncodings={positionalEncodings}
          nEmbd={config.n_embd}
          nVocab={config.n_vocab}
        />
      ) : (
        <>
          {/* Tokens Display */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">Token 序列:</h4>
            <div className="flex flex-wrap gap-2">
              {tokenTexts.map((token, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium"
                >
                  {token}
                </span>
              ))}
            </div>
          </div>

          {/* Current Step Data */}
          {currentStepData && (
            <div className="space-y-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">当前步骤数据:</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex">
                    <span className="font-medium text-gray-600 w-32">步骤类型:</span>
                    <span className="text-gray-800">{currentStepData.step_type}</span>
                  </div>
                  <div className="flex">
                    <span className="font-medium text-gray-600 w-32">层索引:</span>
                    <span className="text-gray-800">{currentStepData.layer_index}</span>
                  </div>
                  <div className="flex">
                    <span className="font-medium text-gray-600 w-32">描述:</span>
                    <span className="text-gray-800">{currentStepData.description}</span>
                  </div>
                </div>
              </div>

              {/* Data Preview */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-gray-700 mb-2">数据预览 (JSON):</h4>
                <div className="bg-gray-900 text-green-400 rounded p-3 overflow-auto max-h-96 text-xs font-mono">
                  <pre>{JSON.stringify(currentStepData, null, 2)}</pre>
                </div>
              </div>

              {/* Metadata */}
              {currentStepData.metadata && Object.keys(currentStepData.metadata).length > 0 && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">元数据:</h4>
                  <div className="bg-gray-900 text-green-400 rounded p-3 overflow-auto max-h-48 text-xs font-mono">
                    <pre>{JSON.stringify(currentStepData.metadata, null, 2)}</pre>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
