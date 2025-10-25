import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { MoEData } from '../types';
import { Heatmap } from './Heatmap';

interface MoEInspectorProps {
  data: MoEData;
  tokens?: string[];
}

export const MoEInspector: React.FC<MoEInspectorProps> = ({ data, tokens }) => {
  const [selectedExpert, setSelectedExpert] = useState<number>(0);
  const [selectedToken, setSelectedToken] = useState<number>(0);

  if (!data) {
    return null;
  }

  const numExperts = data.numExperts || 8;
  const topK = data.topK || 2;
  const tokenLabels = tokens || [];

  const renderGatingWeights = () => {
    if (!data.gatingWeights) {
      return <div className="text-gray-500 text-sm">No gating weights available</div>;
    }

    return (
      <div className="space-y-4">
        <Heatmap
          data={data.gatingWeights}
          title="Gating Weights (Token → Expert Routing)"
          rowLabels={tokenLabels}
          colLabels={Array.from({ length: numExperts }, (_, i) => `E${i}`)}
        />

        <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 text-sm text-purple-900">
          <div className="font-semibold mb-1">💡 Gating Weights</div>
          <p>
            Each row shows the routing probabilities for a token across all experts. The gating network selects the
            top-{topK} experts per token. Brighter colors indicate higher routing probability.
          </p>
        </div>
      </div>
    );
  };

  const renderSelectedExperts = () => {
    if (!data.selectedExperts) {
      return <div className="text-gray-500 text-sm">No expert selection data available</div>;
    }

    return (
      <div className="space-y-3">
        <h6 className="text-sm font-semibold text-gray-700">Selected Experts per Token</h6>

        <div className="flex items-center gap-4">
          <label className="text-sm font-semibold text-gray-700">Token:</label>
          <div className="flex gap-2 flex-wrap">
            {data.selectedExperts.map((_, tokenIdx) => (
              <button
                key={tokenIdx}
                onClick={() => setSelectedToken(tokenIdx)}
                className={`px-3 py-1 text-sm rounded transition-colors ${
                  selectedToken === tokenIdx
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {tokenLabels[tokenIdx] || `T${tokenIdx}`}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg">
          <div className="grid grid-cols-4 gap-4">
            {Array.from({ length: numExperts }, (_, expertId) => {
              const isSelected = data.selectedExperts![selectedToken]?.includes(expertId);
              const gatingWeight = data.gatingWeights?.[selectedToken]?.[expertId];

              return (
                <motion.div
                  key={expertId}
                  className={`p-3 rounded border-2 transition-all ${
                    isSelected
                      ? 'border-purple-500 bg-purple-100'
                      : 'border-gray-300 bg-white opacity-40'
                  }`}
                  whileHover={{ scale: 1.05 }}
                >
                  <div className="text-center">
                    <div className="text-sm font-bold text-gray-800">Expert {expertId}</div>
                    {gatingWeight !== undefined && (
                      <div className="text-xs text-gray-600 mt-1">
                        Weight: {(gatingWeight * 100).toFixed(1)}%
                      </div>
                    )}
                    {isSelected && (
                      <div className="text-xs text-purple-700 mt-1 font-semibold">✓ Active</div>
                    )}
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

        <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 text-sm text-purple-900">
          <div className="font-semibold mb-1">💡 Expert Selection</div>
          <p>
            The gating network selects the top-{topK} experts for each token (highlighted with purple borders). Only
            selected experts contribute to the final output, making the model more efficient and specialized.
          </p>
        </div>
      </div>
    );
  };

  const renderExpertActivations = () => {
    if (!data.expertActivations || data.expertActivations.length === 0) {
      return <div className="text-gray-500 text-sm">No expert activation data available</div>;
    }

    const expertData = data.expertActivations.find((e) => e.expertId === selectedExpert);
    if (!expertData) {
      return <div className="text-gray-500 text-sm">No data for selected expert</div>;
    }

    const activations = expertData.activations;
    const bins = 20;
    const min = Math.min(...activations);
    const max = Math.max(...activations);
    const binWidth = (max - min) / bins;

    const histogram = new Array(bins).fill(0);
    activations.forEach((value) => {
      const binIndex = Math.min(Math.floor((value - min) / binWidth), bins - 1);
      histogram[binIndex]++;
    });

    const maxCount = Math.max(...histogram);
    const mean = activations.reduce((a, b) => a + b, 0) / activations.length;
    const variance = activations.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / activations.length;
    const std = Math.sqrt(variance);

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-4">
          <label className="text-sm font-semibold text-gray-700">Expert:</label>
          <div className="flex gap-2 flex-wrap">
            {data.expertActivations.map((expert) => (
              <button
                key={expert.expertId}
                onClick={() => setSelectedExpert(expert.expertId)}
                className={`px-3 py-1 text-sm rounded transition-colors ${
                  selectedExpert === expert.expertId
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                Expert {expert.expertId}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-gray-50 p-4 rounded-lg">
          <h6 className="text-sm font-semibold text-gray-700 mb-3">
            Feed-Forward Activations Distribution
          </h6>

          <div className="flex gap-4 mb-3 text-xs text-gray-600">
            <div>
              <span className="font-semibold">Min:</span> {min.toFixed(3)}
            </div>
            <div>
              <span className="font-semibold">Max:</span> {max.toFixed(3)}
            </div>
            <div>
              <span className="font-semibold">Mean:</span> {mean.toFixed(3)}
            </div>
            <div>
              <span className="font-semibold">Std:</span> {std.toFixed(3)}
            </div>
          </div>

          <div className="flex items-end gap-1 h-48">
            {histogram.map((count, idx) => {
              const heightPercent = maxCount > 0 ? (count / maxCount) * 100 : 0;
              const binStart = min + idx * binWidth;

              return (
                <motion.div
                  key={idx}
                  className="flex-1 bg-purple-500 rounded-t relative group cursor-pointer"
                  style={{ height: `${heightPercent}%` }}
                  initial={{ height: 0 }}
                  animate={{ height: `${heightPercent}%` }}
                  transition={{ duration: 0.3, delay: idx * 0.02 }}
                  whileHover={{ opacity: 0.8 }}
                >
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                    <div>Range: [{binStart.toFixed(2)}, {(binStart + binWidth).toFixed(2)})</div>
                    <div>Count: {count}</div>
                  </div>
                </motion.div>
              );
            })}
          </div>

          <div className="mt-2 text-xs text-gray-500 text-center">Activation Values</div>
        </div>

        <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 text-sm text-purple-900">
          <div className="font-semibold mb-1">💡 Expert Activations</div>
          <p>
            This histogram shows the distribution of feed-forward activation values for Expert {selectedExpert}. Each
            expert learns specialized patterns, resulting in different activation distributions. Hover over bars for
            details.
          </p>
        </div>
      </div>
    );
  };

  const renderMetadata = () => {
    const numTokens = data.gatingWeights?.length || 0;

    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h6 className="text-sm font-semibold text-gray-800 mb-3">MoE Metadata</h6>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-600">Number of Experts:</span>
            <span className="ml-2 font-semibold text-gray-800">{numExperts}</span>
          </div>
          <div>
            <span className="text-gray-600">Top-K Selection:</span>
            <span className="ml-2 font-semibold text-gray-800">{topK}</span>
          </div>
          <div>
            <span className="text-gray-600">Num Tokens:</span>
            <span className="ml-2 font-semibold text-gray-800">{numTokens}</span>
          </div>
          <div>
            <span className="text-gray-600">Active Rate:</span>
            <span className="ml-2 font-semibold text-gray-800">
              {((topK / numExperts) * 100).toFixed(1)}%
            </span>
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
      <h5 className="text-base font-bold text-gray-800">Mixture of Experts Inspector</h5>

      {renderMetadata()}
      {renderGatingWeights()}
      {renderSelectedExperts()}
      {renderExpertActivations()}
    </motion.div>
  );
};
