'use client';

import { useState } from 'react';
import { AppLayout } from '@/components/layout/AppLayout';
import { Matrix3D } from '@/components/visualization/matrix/Matrix3D';
import { AttentionMatrix } from '@/components/visualization/attention/AttentionMatrix';
import { FormulaDisplay, FORMULAS, FORMULA_VARIABLES } from '@/components/explanation/FormulaDisplay';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Play, Pause } from 'lucide-react';

export default function VizShowcasePage() {
  const [is3DPlaying, setIs3DPlaying] = useState(false);
  
  // Sample data
  const tokens = ['The', 'quick', 'brown', 'fox', 'jumps'];
  const attentionData = generateAttentionMatrix(tokens.length);
  const embeddingData = generateEmbeddingMatrix(tokens.length, 8);
  
  return (
    <AppLayout showSidebar={false}>
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Hero */}
        <div className="text-center space-y-4 py-12">
          <h1 className="text-5xl font-bold text-white">
            Visualization Showcase
          </h1>
          <p className="text-xl text-slate-400 max-w-3xl mx-auto">
            Explore our refactored visualization system with 3D matrices, 
            interactive attention maps, and mathematical formulas
          </p>
        </div>
        
        {/* Main Content */}
        <Tabs defaultValue="3d" className="w-full">
          <TabsList className="grid w-full grid-cols-3 max-w-md mx-auto bg-slate-800">
            <TabsTrigger value="3d">3D Matrices</TabsTrigger>
            <TabsTrigger value="2d">2D Attention</TabsTrigger>
            <TabsTrigger value="formulas">Formulas</TabsTrigger>
          </TabsList>
          
          <TabsContent value="3d" className="space-y-6 mt-8">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">3D Matrix Visualization</CardTitle>
                <CardDescription>
                  Interactive Three.js visualization with orbit controls. 
                  Drag to rotate, scroll to zoom.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Matrix3D
                    data={attentionData}
                    title="Attention Weights (3D)"
                    showValues={false}
                    interactive={true}
                    onCellClick={(i, j, value) => 
                      console.log(`Cell [${i}, ${j}] = ${value}`)
                    }
                  />
                  
                  <Matrix3D
                    data={embeddingData}
                    title="Token Embeddings (3D)"
                    showValues={false}
                    interactive={true}
                  />
                </div>
                
                <div className="bg-slate-900 border border-slate-700 rounded-lg p-4">
                  <h4 className="text-white font-medium mb-2">Features:</h4>
                  <ul className="text-slate-300 text-sm space-y-1 list-disc list-inside">
                    <li>Real-time 3D rendering with Three.js</li>
                    <li>Interactive camera controls (rotate, zoom, pan)</li>
                    <li>Hover highlighting for rows and columns</li>
                    <li>Click handling for cell selection</li>
                    <li>Configurable colors and scaling</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="2d" className="space-y-6 mt-8">
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">2D Attention Visualization</CardTitle>
                <CardDescription>
                  D3.js-powered heatmap with rich interactions and smooth animations.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <AttentionMatrix
                  attentionWeights={attentionData}
                  tokens={tokens}
                  title="Self-Attention Heatmap"
                  showValues={false}
                  animated={true}
                  onHover={(i, j, value) => 
                    console.log(`Attention from "${tokens[i]}" to "${tokens[j]}": ${value.toFixed(4)}`)
                  }
                />
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <AttentionMatrix
                    attentionWeights={embeddingData.slice(0, 5)}
                    tokens={tokens}
                    title="Query Matrix"
                    showValues={true}
                    animated={false}
                  />
                  
                  <AttentionMatrix
                    attentionWeights={embeddingData.slice(0, 5)}
                    tokens={tokens}
                    title="Key Matrix"
                    showValues={true}
                    animated={false}
                  />
                </div>
                
                <div className="bg-slate-900 border border-slate-700 rounded-lg p-4">
                  <h4 className="text-white font-medium mb-2">Features:</h4>
                  <ul className="text-slate-300 text-sm space-y-1 list-disc list-inside">
                    <li>SVG-based D3.js rendering for crisp visuals</li>
                    <li>Hover to highlight rows and columns</li>
                    <li>Smooth entrance animations with stagger</li>
                    <li>Color scales using viridis/plasma schemes</li>
                    <li>Optional value display for precise inspection</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="formulas" className="space-y-6 mt-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <FormulaDisplay
                formula={FORMULAS.attention}
                title="Self-Attention"
                explanation="The core mechanism that allows tokens to attend to each other based on learned query, key, and value representations."
                variables={FORMULA_VARIABLES.attention}
              />
              
              <FormulaDisplay
                formula={FORMULAS.multiHeadAttention}
                title="Multi-Head Attention"
                explanation="Extends self-attention by applying it in parallel across multiple representation subspaces."
                variables={FORMULA_VARIABLES.multiHeadAttention}
              />
              
              <FormulaDisplay
                formula={FORMULAS.layerNorm}
                title="Layer Normalization"
                explanation="Normalizes activations to stabilize training and improve convergence."
                variables={FORMULA_VARIABLES.layerNorm}
              />
              
              <FormulaDisplay
                formula={FORMULAS.feedForward}
                title="Feed-Forward Network"
                explanation="Two-layer fully connected network applied to each position independently."
                variables={FORMULA_VARIABLES.feedForward}
              />
              
              <FormulaDisplay
                formula={FORMULAS.positionalEncoding}
                title="Positional Encoding"
                explanation="Injects position information using sinusoidal functions of different frequencies."
                variables={FORMULA_VARIABLES.positionalEncoding}
              />
              
              <FormulaDisplay
                formula={FORMULAS.moeGating}
                title="MoE Gating"
                explanation="Routes inputs to specialized expert networks based on learned gating scores."
                variables={FORMULA_VARIABLES.moe}
              />
            </div>
            
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white">LaTeX Support</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-slate-900 border border-slate-700 rounded-lg p-4">
                  <h4 className="text-white font-medium mb-2">Features:</h4>
                  <ul className="text-slate-300 text-sm space-y-1 list-disc list-inside">
                    <li>Fast math typesetting with KaTeX</li>
                    <li>Support for display and inline modes</li>
                    <li>Pre-defined formulas for common operations</li>
                    <li>Variable explanations with inline math</li>
                    <li>Dark theme optimized rendering</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
        
        {/* Technology Stack */}
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-white">Technology Stack</CardTitle>
            <CardDescription>
              Built with modern web technologies for optimal performance and user experience.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { name: 'Three.js', desc: '3D Graphics' },
                { name: 'D3.js', desc: 'Data Viz' },
                { name: 'GSAP', desc: 'Animations' },
                { name: 'KaTeX', desc: 'Math Formulas' },
                { name: 'React 19', desc: 'UI Framework' },
                { name: 'Next.js 16', desc: 'App Framework' },
                { name: 'Tailwind CSS', desc: 'Styling' },
                { name: 'shadcn/ui', desc: 'Components' },
              ].map((tech) => (
                <div key={tech.name} className="bg-slate-900 rounded-lg p-4 text-center">
                  <div className="font-semibold text-white">{tech.name}</div>
                  <div className="text-sm text-slate-400">{tech.desc}</div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        
        {/* Call to Action */}
        <div className="text-center py-12 space-y-4">
          <h2 className="text-3xl font-bold text-white">
            Ready to explore Transformer architecture?
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Visit our interactive demo to see these visualizations in action with real computation steps.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Button asChild size="lg" className="gradient-purple-pink">
              <a href="/transformer-viz">
                <Play className="w-5 h-5 mr-2" />
                Try Interactive Demo
              </a>
            </Button>
            <Button asChild size="lg" variant="outline">
              <a href="/">
                Back to Home
              </a>
            </Button>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}

// Helper functions
function generateAttentionMatrix(size: number): number[][] {
  const matrix: number[][] = [];
  for (let i = 0; i < size; i++) {
    const row: number[] = [];
    for (let j = 0; j < size; j++) {
      const distance = Math.abs(i - j);
      const value = Math.exp(-distance / 2) + Math.random() * 0.1;
      row.push(value);
    }
    const sum = row.reduce((a, b) => a + b, 0);
    matrix.push(row.map(v => v / sum));
  }
  return matrix;
}

function generateEmbeddingMatrix(tokens: number, dims: number): number[][] {
  const matrix: number[][] = [];
  for (let i = 0; i < tokens; i++) {
    const row: number[] = [];
    for (let j = 0; j < dims; j++) {
      row.push(Math.random() * 2 - 1);
    }
    matrix.push(row);
  }
  return matrix;
}
