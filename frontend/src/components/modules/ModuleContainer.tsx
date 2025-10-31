'use client';

import React from 'react';
import { X, Maximize2, Minimize2, Settings } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { cn } from '@/lib/design-system';
import { useVisualization } from '@/contexts/VisualizationContext';

interface ModuleContainerProps {
  type: 'embedding' | 'attention' | 'moe' | 'output';
  className?: string;
}

const moduleConfig = {
  embedding: {
    title: 'Token Embedding',
    color: 'from-indigo-500 to-blue-600',
    description: 'Convert tokens to vectors with positional encoding',
  },
  attention: {
    title: 'Multi-Head Attention',
    color: 'from-emerald-500 to-green-600',
    description: 'Self-attention mechanism with multiple heads',
  },
  moe: {
    title: 'Mixture of Experts',
    color: 'from-purple-500 to-pink-600',
    description: 'Sparse feed-forward network with expert routing',
  },
  output: {
    title: 'Output Layer',
    color: 'from-amber-500 to-orange-600',
    description: 'Generate predictions and probabilities',
  },
};

export function ModuleContainer({ type, className }: ModuleContainerProps) {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const { state: vizState, actions: vizActions } = useVisualization();
  const config = moduleConfig[type];
  const isActive = vizState.modules[type].isActive;

  if (!isActive) return null;

  return (
    <Card
      className={cn(
        "absolute bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm border shadow-lg transition-all duration-300",
        isExpanded ? "w-96 h-64" : "w-64 h-48",
        type === 'embedding' && "top-4 left-4",
        type === 'attention' && "top-4 right-4",
        type === 'moe' && "bottom-4 left-4",
        type === 'output' && "bottom-4 right-4",
        className
      )}
    >
      {/* Header */}
      <div className={cn(
        "bg-gradient-to-r p-3 rounded-t-lg flex items-center justify-between",
        config.color
      )}>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
          <h3 className="text-white font-semibold text-sm">{config.title}</h3>
        </div>
        <div className="flex items-center space-x-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1 h-auto text-white hover:bg-white/20"
          >
            {isExpanded ? (
              <Minimize2 className="w-3 h-3" />
            ) : (
              <Maximize2 className="w-3 h-3" />
            )}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => vizActions.updateModule(type, { isActive: false })}
            className="p-1 h-auto text-white hover:bg-white/20"
          >
            <X className="w-3 h-3" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="p-4 h-full">
        <p className="text-xs text-slate-600 dark:text-slate-400 mb-3">
          {config.description}
        </p>

        {/* Module-specific content */}
        <div className="space-y-2">
          {type === 'embedding' && (
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
                Tokens: {vizState.modules.embedding.tokens.length}
              </div>
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
                Embedding Dim: {vizState.modules.embedding.embeddings[0]?.length || 768}
              </div>
              {isExpanded && (
                <div className="text-xs text-slate-600 dark:text-slate-400">
                  <div className="font-medium mb-1">Sample tokens:</div>
                  <div className="flex flex-wrap gap-1">
                    {vizState.modules.embedding.tokens.slice(0, 5).map((token, i) => (
                      <span
                        key={i}
                        className="px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded text-xs"
                      >
                        {token}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {type === 'attention' && (
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
                Heads: {vizState.modules.attention.heads}
              </div>
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
                Attention Weights: {vizState.modules.attention.attentionWeights.length} layers
              </div>
              {isExpanded && (
                <div className="text-xs text-slate-600 dark:text-slate-400">
                  <div className="font-medium mb-1">Attention Pattern:</div>
                  <div className="grid grid-cols-4 gap-1">
                    {Array.from({ length: 16 }, (_, i) => (
                      <div
                        key={i}
                        className="w-4 h-4 bg-gradient-to-br from-blue-400 to-purple-500 rounded"
                        style={{ opacity: Math.random() * 0.8 + 0.2 }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {type === 'moe' && (
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
                Experts: {vizState.modules.moe.experts}
              </div>
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
                Top-k: 2
              </div>
              {isExpanded && (
                <div className="text-xs text-slate-600 dark:text-slate-400">
                  <div className="font-medium mb-1">Expert Usage:</div>
                  <div className="space-y-1">
                    {Array.from({ length: Math.min(4, vizState.modules.moe.experts) }, (_, i) => (
                      <div key={i} className="flex items-center space-x-2">
                        <span className="w-16">Expert {i + 1}:</span>
                        <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-purple-400 to-pink-500 h-2 rounded-full"
                            style={{ width: `${Math.random() * 80 + 20}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {type === 'output' && (
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
                Vocabulary Size: 50,257
              </div>
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">
                Top Predictions: {vizState.modules.output.predictions.length}
              </div>
              {isExpanded && (
                <div className="text-xs text-slate-600 dark:text-slate-400">
                  <div className="font-medium mb-1">Top Tokens:</div>
                  <div className="space-y-1">
                    {vizState.modules.output.predictions.slice(0, 3).map((pred, i) => (
                      <div key={i} className="flex items-center justify-between">
                        <span>{pred.token}</span>
                        <span className="font-mono">
                          {(pred.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}