'use client';

import { useState } from 'react';
import { useVisualizationStore } from '@/store/visualizationStore';
import { TokenEmbeddingVisualization } from './visualizations';
import { Card, EmptyState } from './ui';
import { Eye, Code } from 'lucide-react';

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
      <Card className="min-h-[500px] flex items-center justify-center">
        <EmptyState />
      </Card>
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
    <Card className="min-h-[500px] relative overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 opacity-5 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-500 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-pink-500 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <div className="relative z-10 flex items-center justify-between mb-6">
        <div className="text-center flex-1">
          {currentStepData && (
            <>
              <div className="inline-block bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/30 rounded-full px-4 py-1 mb-2">
                <span className="text-sm text-purple-300">当前步骤</span>
              </div>
              <h2 className="text-2xl font-bold text-white mb-1">
                {currentStepData.step_type}
              </h2>
              <p className="text-slate-400 text-sm">
                {currentStepData.description}
              </p>
            </>
          )}
        </div>
        
        {/* View Toggle */}
        <div className="flex gap-2 bg-slate-900/50 p-1 rounded-lg border border-slate-700/50">
          <button
            onClick={() => setShowD3Viz(true)}
            className={`px-3 py-1.5 text-sm rounded transition-all flex items-center gap-1.5 ${
              showD3Viz
                ? 'bg-purple-500 text-white shadow-lg shadow-purple-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="w-4 h-4" />
            动画视图
          </button>
          <button
            onClick={() => setShowD3Viz(false)}
            className={`px-3 py-1.5 text-sm rounded transition-all flex items-center gap-1.5 ${
              !showD3Viz
                ? 'bg-purple-500 text-white shadow-lg shadow-purple-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Code className="w-4 h-4" />
            数据视图
          </button>
        </div>
      </div>

      {showD3Viz && embeddings.length > 0 ? (
        <div className="relative z-10 bg-slate-900/30 rounded-lg p-4 border border-slate-700/50">
          <TokenEmbeddingVisualization
            text={inputText}
            tokens={tokens}
            tokenTexts={tokenTexts}
            embeddings={embeddings}
            positionalEncodings={positionalEncodings}
            nEmbd={config.n_embd}
            nVocab={config.n_vocab}
          />
        </div>
      ) : (
        <div className="relative z-10">
          {/* Tokens Display */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-slate-300 mb-3">Token 序列:</h4>
            <div className="flex flex-wrap gap-2">
              {tokenTexts.map((token, idx) => (
                <span
                  key={idx}
                  className="px-3 py-1.5 bg-gradient-to-r from-purple-500/20 to-pink-500/20 text-purple-200 rounded-lg text-sm font-medium border border-purple-500/30"
                >
                  {token}
                </span>
              ))}
            </div>
          </div>

          {/* Current Step Data */}
          {currentStepData && (
            <div className="space-y-4">
              <div className="bg-slate-900/30 rounded-lg p-4 border border-slate-700/50">
                <h4 className="text-sm font-semibold text-slate-300 mb-3">当前步骤信息:</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex">
                    <span className="font-medium text-slate-400 w-32">步骤类型:</span>
                    <span className="text-white">{currentStepData.step_type}</span>
                  </div>
                  <div className="flex">
                    <span className="font-medium text-slate-400 w-32">层索引:</span>
                    <span className="text-white">{currentStepData.layer_index}</span>
                  </div>
                  <div className="flex">
                    <span className="font-medium text-slate-400 w-32">描述:</span>
                    <span className="text-white">{currentStepData.description}</span>
                  </div>
                </div>
              </div>

              {/* Data Preview */}
              <div className="bg-slate-900/30 rounded-lg p-4 border border-slate-700/50">
                <h4 className="text-sm font-semibold text-slate-300 mb-3">数据预览 (JSON):</h4>
                <div className="bg-slate-950 text-emerald-400 rounded-lg p-3 overflow-auto max-h-96 text-xs font-mono border border-slate-800">
                  <pre>{JSON.stringify(currentStepData, null, 2)}</pre>
                </div>
              </div>

              {/* Metadata */}
              {currentStepData.metadata && Object.keys(currentStepData.metadata).length > 0 && (
                <div className="bg-slate-900/30 rounded-lg p-4 border border-slate-700/50">
                  <h4 className="text-sm font-semibold text-slate-300 mb-3">元数据:</h4>
                  <div className="bg-slate-950 text-emerald-400 rounded-lg p-3 overflow-auto max-h-48 text-xs font-mono border border-slate-800">
                    <pre>{JSON.stringify(currentStepData.metadata, null, 2)}</pre>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Legend */}
      {showD3Viz && embeddings.length > 0 && (
        <div className="relative z-10 mt-4 flex flex-wrap gap-3 justify-center">
          <div className="flex items-center gap-2 text-sm">
            <div className="w-4 h-4 bg-purple-500 rounded" />
            <span className="text-slate-300">Token嵌入</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <div className="w-4 h-4 bg-pink-500 rounded" />
            <span className="text-slate-300">位置编码</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <div className="w-4 h-4 bg-cyan-500 rounded" />
            <span className="text-slate-300">最终向量</span>
          </div>
        </div>
      )}
    </Card>
  );
}
