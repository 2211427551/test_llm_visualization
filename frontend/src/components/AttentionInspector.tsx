import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { AttentionData } from '../types';
import { Heatmap } from './Heatmap';

interface AttentionInspectorProps {
  data: AttentionData;
  tokens?: string[];
}

type ViewMode = 'attention' | 'qkv';

export const AttentionInspector: React.FC<AttentionInspectorProps> = ({ data, tokens }) => {
  const [viewMode, setViewMode] = useState<ViewMode>('attention');
  const [selectedHead, setSelectedHead] = useState<number>(0);

  if (!data) {
    return null;
  }

  const numHeads = data.numHeads || 1;
  const headDim = data.headDim || 64;
  const tokenLabels = tokens || [];

  const renderAttentionScores = () => {
    if (!data.attentionScores || !data.sparsityMask) {
      return <div className="text-gray-500 text-sm">No attention scores available</div>;
    }

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <label className="text-sm font-semibold text-gray-700">Attention Head:</label>
          <div className="flex gap-2 flex-wrap">
            {Array.from({ length: numHeads }, (_, i) => (
              <button
                key={i}
                onClick={() => setSelectedHead(i)}
                className={`px-3 py-1 text-sm rounded transition-colors ${
                  selectedHead === i
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Head {i}
              </button>
            ))}
          </div>
        </div>

        <Heatmap
          data={data.attentionScores}
          mask={data.sparsityMask}
          title={`Attention Scores - Head ${selectedHead}`}
          rowLabels={tokenLabels}
          colLabels={tokenLabels}
        />

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-900">
          <div className="font-semibold mb-1">💡 Attention Visualization</div>
          <p>
            Each row shows which tokens the current token attends to. Brighter colors indicate stronger attention.
            Gray cells represent pruned (masked) connections where attention is forced to zero for sparsity.
          </p>
        </div>
      </div>
    );
  };

  const renderQKVMatrices = () => {
    const hasQuery = data.queryMatrix && data.queryMatrix.length > 0;
    const hasKey = data.keyMatrix && data.keyMatrix.length > 0;
    const hasValue = data.valueMatrix && data.valueMatrix.length > 0;

    if (!hasQuery && !hasKey && !hasValue) {
      return <div className="text-gray-500 text-sm">No Q/K/V matrices available</div>;
    }

    return (
      <div className="space-y-4">
        {hasQuery && (
          <Heatmap
            data={data.queryMatrix!}
            title="Query (Q) Matrix"
            rowLabels={tokenLabels}
            colLabels={Array.from({ length: numHeads * headDim }, (_, i) => `d${i}`)}
          />
        )}

        {hasKey && (
          <Heatmap
            data={data.keyMatrix!}
            title="Key (K) Matrix"
            rowLabels={tokenLabels}
            colLabels={Array.from({ length: numHeads * headDim }, (_, i) => `d${i}`)}
          />
        )}

        {hasValue && (
          <Heatmap
            data={data.valueMatrix!}
            title="Value (V) Matrix"
            rowLabels={tokenLabels}
            colLabels={Array.from({ length: numHeads * headDim }, (_, i) => `d${i}`)}
          />
        )}

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm text-blue-900">
          <div className="font-semibold mb-1">💡 Q/K/V Matrices</div>
          <p>
            <strong>Query (Q):</strong> What each token is looking for.{' '}
            <strong>Key (K):</strong> What each token offers.{' '}
            <strong>Value (V):</strong> The actual information to be aggregated.{' '}
            Attention scores are computed as softmax(Q·K<sup>T</sup>), then multiplied with V.
          </p>
        </div>
      </div>
    );
  };

  const renderMetadata = () => {
    const numTokens = data.attentionScores?.length || 0;
    const totalConnections = numTokens * numTokens;
    const maskedConnections =
      data.sparsityMask?.reduce((sum, row) => sum + row.filter((v) => v === 0).length, 0) || 0;
    const sparsityPercent = totalConnections > 0 ? ((maskedConnections / totalConnections) * 100).toFixed(1) : '0.0';

    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h6 className="text-sm font-semibold text-gray-800 mb-3">Attention Metadata</h6>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-600">Number of Heads:</span>
            <span className="ml-2 font-semibold text-gray-800">{numHeads}</span>
          </div>
          <div>
            <span className="text-gray-600">Head Dimension:</span>
            <span className="ml-2 font-semibold text-gray-800">{headDim}</span>
          </div>
          <div>
            <span className="text-gray-600">Embedding Dim:</span>
            <span className="ml-2 font-semibold text-gray-800">{numHeads * headDim}</span>
          </div>
          <div>
            <span className="text-gray-600">Num Tokens:</span>
            <span className="ml-2 font-semibold text-gray-800">{numTokens}</span>
          </div>
          <div>
            <span className="text-gray-600">Total Connections:</span>
            <span className="ml-2 font-semibold text-gray-800">{totalConnections}</span>
          </div>
          <div>
            <span className="text-gray-600">Sparsity:</span>
            <span className="ml-2 font-semibold text-gray-800">{sparsityPercent}%</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-4"
    >
      <div className="flex items-center justify-between">
        <h5 className="text-base font-bold text-gray-800">Attention Inspector</h5>
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('attention')}
            className={`px-4 py-2 text-sm rounded transition-colors ${
              viewMode === 'attention'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Attention Scores
          </button>
          <button
            onClick={() => setViewMode('qkv')}
            className={`px-4 py-2 text-sm rounded transition-colors ${
              viewMode === 'qkv' ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Q/K/V Matrices
          </button>
        </div>
      </div>

      {renderMetadata()}

      {viewMode === 'attention' ? renderAttentionScores() : renderQKVMatrices()}
    </motion.div>
  );
};
