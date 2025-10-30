'use client';

import { PlaybackControls } from '@/components/controls/PlaybackControls';
import { FormulaDisplay, FORMULAS, FORMULA_VARIABLES } from '@/components/explanation/FormulaDisplay';
import { usePlaybackStore } from '@/stores/playback-store';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Sidebar() {
  const { currentStep, steps } = usePlaybackStore();
  const currentStepData = steps[currentStep];
  
  return (
    <div className="p-4 space-y-4">
      <Tabs defaultValue="controls" className="w-full">
        <TabsList className="grid w-full grid-cols-2 bg-slate-800">
          <TabsTrigger value="controls">Controls</TabsTrigger>
          <TabsTrigger value="formula">Formula</TabsTrigger>
        </TabsList>
        
        <TabsContent value="controls" className="space-y-4 mt-4">
          <PlaybackControls />
          
          {currentStepData && (
            <Card className="bg-slate-800 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white text-sm">Step Details</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {currentStepData.description}
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="formula" className="mt-4">
          <FormulaDisplay
            formula={currentStepData?.formula || FORMULAS.attention}
            title="Current Step Formula"
            explanation={getFormulaExplanation(currentStepData?.id)}
            variables={getFormulaVariables(currentStepData?.id)}
          />
        </TabsContent>
      </Tabs>
      
      {/* Layer info */}
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-white text-sm">Architecture Info</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">Layers:</span>
            <span className="text-white font-mono">12</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Hidden Size:</span>
            <span className="text-white font-mono">768</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Attention Heads:</span>
            <span className="text-white font-mono">12</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Vocab Size:</span>
            <span className="text-white font-mono">50,257</span>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function getFormulaExplanation(stepId?: string): string {
  const explanations: Record<string, string> = {
    'embedding': 'Each token is converted into a dense vector representation.',
    'positional': 'Position information is added to embeddings using sinusoidal functions.',
    'attention': 'Computes attention weights between all token pairs.',
    'multihead': 'Applies attention mechanism in parallel across multiple heads.',
    'layernorm': 'Normalizes activations to stabilize training.',
    'ffn': 'Applies a two-layer feed-forward network with ReLU activation.',
    'moe': 'Routes inputs to specialized expert networks based on learned gating.',
  };
  
  return explanations[stepId || ''] || 'Transform input through various layers.';
}

function getFormulaVariables(stepId?: string): Record<string, string> {
  const variableMap: Record<string, Record<string, string>> = {
    'attention': FORMULA_VARIABLES.attention,
    'multihead': FORMULA_VARIABLES.multiHeadAttention,
    'layernorm': FORMULA_VARIABLES.layerNorm,
    'ffn': FORMULA_VARIABLES.feedForward,
    'positional': FORMULA_VARIABLES.positionalEncoding,
    'moe': FORMULA_VARIABLES.moe,
  };
  
  return variableMap[stepId || ''] || {};
}
