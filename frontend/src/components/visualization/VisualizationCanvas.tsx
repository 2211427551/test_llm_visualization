'use client';

import { useState, useEffect } from 'react';
import { Matrix3D } from './matrix/Matrix3D';
import { AttentionMatrix } from './attention/AttentionMatrix';
import { usePlaybackStore } from '@/stores/playback-store';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface VisualizationCanvasProps {
  mode?: '2d' | '3d' | 'auto';
  data?: {
    tokens?: string[];
    embeddings?: number[][];
    attention?: number[][];
  };
}

export function VisualizationCanvas({ 
  mode = 'auto',
  data = {}
}: VisualizationCanvasProps) {
  const { currentStep, steps } = usePlaybackStore();
  const [visualizationMode, setVisualizationMode] = useState<'2d' | '3d'>(mode === 'auto' ? '2d' : mode);
  
  // Sample data for demonstration
  const sampleTokens = data.tokens || ['The', 'cat', 'sat', 'on', 'mat'];
  const sampleAttention = data.attention || generateSampleAttention(sampleTokens.length);
  const sampleEmbeddings = data.embeddings || generateSampleEmbeddings(sampleTokens.length, 8);
  
  const currentStepData = steps[currentStep];
  
  useEffect(() => {
    // Auto-switch visualization mode based on step
    if (mode === 'auto' && currentStepData) {
      if (currentStepData.id.includes('3d') || currentStepData.id.includes('matrix')) {
        setVisualizationMode('3d');
      } else {
        setVisualizationMode('2d');
      }
    }
  }, [currentStepData, mode]);
  
  return (
    <div className="w-full space-y-6">
      {/* Mode switcher */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">
          {currentStepData?.name || 'Transformer Visualization'}
        </h2>
        
        <Tabs 
          value={visualizationMode} 
          onValueChange={(v) => setVisualizationMode(v as '2d' | '3d')}
          className="w-auto"
        >
          <TabsList className="bg-slate-800">
            <TabsTrigger value="2d">2D View</TabsTrigger>
            <TabsTrigger value="3d">3D View</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      
      {/* Visualization content */}
      <div className="min-h-[600px]">
        {visualizationMode === '3d' ? (
          <div className="space-y-6">
            <Matrix3D
              data={sampleEmbeddings}
              title="Token Embeddings (3D)"
              showValues={false}
              interactive={true}
            />
            
            <Matrix3D
              data={sampleAttention}
              title="Attention Weights (3D)"
              showValues={false}
              interactive={true}
            />
          </div>
        ) : (
          <div className="space-y-6">
            <AttentionMatrix
              attentionWeights={sampleAttention}
              tokens={sampleTokens}
              title="Attention Weights"
              showValues={false}
              animated={true}
            />
            
            <div className="grid grid-cols-2 gap-6">
              <div className="bg-slate-900 rounded-xl p-6">
                <h3 className="text-white font-medium mb-4">Query</h3>
                <AttentionMatrix
                  attentionWeights={sampleEmbeddings}
                  tokens={sampleTokens.slice(0, sampleEmbeddings.length)}
                  showValues={true}
                  animated={false}
                />
              </div>
              
              <div className="bg-slate-900 rounded-xl p-6">
                <h3 className="text-white font-medium mb-4">Key</h3>
                <AttentionMatrix
                  attentionWeights={sampleEmbeddings}
                  tokens={sampleTokens.slice(0, sampleEmbeddings.length)}
                  showValues={true}
                  animated={false}
                />
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Step description */}
      {currentStepData && (
        <div className="bg-gradient-to-r from-purple-900/20 to-pink-900/20 border border-purple-500/20 rounded-xl p-6">
          <h3 className="text-white font-medium mb-2">What's happening?</h3>
          <p className="text-slate-300 leading-relaxed">
            {currentStepData.description}
          </p>
        </div>
      )}
    </div>
  );
}

// Helper functions to generate sample data
function generateSampleAttention(size: number): number[][] {
  const matrix: number[][] = [];
  for (let i = 0; i < size; i++) {
    const row: number[] = [];
    for (let j = 0; j < size; j++) {
      // Create attention pattern: higher values closer to diagonal
      const distance = Math.abs(i - j);
      const value = Math.exp(-distance / 2) + Math.random() * 0.1;
      row.push(value);
    }
    // Normalize row
    const sum = row.reduce((a, b) => a + b, 0);
    matrix.push(row.map(v => v / sum));
  }
  return matrix;
}

function generateSampleEmbeddings(tokens: number, dims: number): number[][] {
  const matrix: number[][] = [];
  for (let i = 0; i < tokens; i++) {
    const row: number[] = [];
    for (let j = 0; j < dims; j++) {
      row.push(Math.random());
    }
    matrix.push(row);
  }
  return matrix;
}
