'use client';

import { useEffect } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { VisualizationCanvas } from '@/components/visualization/VisualizationCanvas';
import { usePlaybackStore } from '@/stores/playback-store';
import { FORMULAS } from '@/components/explanation/FormulaDisplay';

export default function TransformerVizPage() {
  const { setSteps } = usePlaybackStore();
  
  useEffect(() => {
    // Initialize the steps for the visualization
    setSteps([
      {
        id: 'tokenization',
        name: 'Tokenization',
        description: 'Input text is split into tokens. Each word or subword becomes a discrete unit.',
        duration: 2,
      },
      {
        id: 'embedding',
        name: 'Token Embedding',
        description: 'Each token is converted into a dense vector representation of fixed dimension.',
        duration: 2,
        formula: FORMULAS.positionalEncoding,
      },
      {
        id: 'positional',
        name: 'Positional Encoding',
        description: 'Position information is added to embeddings using sinusoidal functions to encode sequence order.',
        duration: 2,
        formula: FORMULAS.positionalEncoding,
      },
      {
        id: 'attention',
        name: 'Self-Attention',
        description: 'Computes Query, Key, and Value matrices, then calculates attention weights between all token pairs.',
        duration: 3,
        formula: FORMULAS.attention,
      },
      {
        id: 'multihead',
        name: 'Multi-Head Attention',
        description: 'Applies attention mechanism in parallel across multiple heads, allowing the model to attend to different aspects.',
        duration: 3,
        formula: FORMULAS.multiHeadAttention,
      },
      {
        id: 'attention-output',
        name: 'Attention Output',
        description: 'Concatenates all attention heads and projects back to model dimension.',
        duration: 2,
      },
      {
        id: 'residual-1',
        name: 'Residual Connection',
        description: 'Adds the input to the attention output to preserve information and ease training.',
        duration: 1,
      },
      {
        id: 'layernorm-1',
        name: 'Layer Normalization',
        description: 'Normalizes activations across the feature dimension to stabilize training.',
        duration: 2,
        formula: FORMULAS.layerNorm,
      },
      {
        id: 'ffn',
        name: 'Feed-Forward Network',
        description: 'Applies a two-layer feed-forward network with ReLU activation to each position independently.',
        duration: 3,
        formula: FORMULAS.feedForward,
      },
      {
        id: 'residual-2',
        name: 'Residual Connection',
        description: 'Another residual connection after the feed-forward network.',
        duration: 1,
      },
      {
        id: 'layernorm-2',
        name: 'Layer Normalization',
        description: 'Final layer normalization in the transformer block.',
        duration: 2,
        formula: FORMULAS.layerNorm,
      },
      {
        id: 'output',
        name: 'Output',
        description: 'The processed representation is ready for the next layer or final prediction.',
        duration: 2,
      },
    ]);
  }, [setSteps]);
  
  return (
    <AppLayout showSidebar={true}>
      <div className="space-y-6">
        {/* Hero section */}
        <div className="text-center space-y-3 py-8">
          <h1 className="text-4xl font-bold text-white">
            Transformer Architecture
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto">
            Interactive visualization of the Transformer model computation process
          </p>
          <div className="flex items-center justify-center gap-4 text-sm text-slate-500">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-purple-500"></div>
              <span>Use controls on the left to navigate</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-pink-500"></div>
              <span>Switch between 2D and 3D views</span>
            </div>
          </div>
        </div>
        
        {/* Main visualization */}
        <VisualizationCanvas mode="auto" />
        
        {/* Info cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <div className="bg-gradient-to-br from-purple-900/20 to-purple-800/20 border border-purple-500/30 rounded-xl p-6">
            <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center mb-4">
              <span className="text-2xl">🧠</span>
            </div>
            <h3 className="text-white font-semibold mb-2">Attention Mechanism</h3>
            <p className="text-slate-400 text-sm">
              Visualize how tokens attend to each other through the self-attention mechanism.
            </p>
          </div>
          
          <div className="bg-gradient-to-br from-pink-900/20 to-pink-800/20 border border-pink-500/30 rounded-xl p-6">
            <div className="w-12 h-12 bg-pink-500 rounded-lg flex items-center justify-center mb-4">
              <span className="text-2xl">📊</span>
            </div>
            <h3 className="text-white font-semibold mb-2">Matrix Operations</h3>
            <p className="text-slate-400 text-sm">
              See the matrix multiplications and transformations in 3D space.
            </p>
          </div>
          
          <div className="bg-gradient-to-br from-cyan-900/20 to-cyan-800/20 border border-cyan-500/30 rounded-xl p-6">
            <div className="w-12 h-12 bg-cyan-500 rounded-lg flex items-center justify-center mb-4">
              <span className="text-2xl">⚡</span>
            </div>
            <h3 className="text-white font-semibold mb-2">Real-time Updates</h3>
            <p className="text-slate-400 text-sm">
              Control the animation speed and step through the computation process.
            </p>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
