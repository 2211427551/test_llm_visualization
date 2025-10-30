'use client';

import { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface FormulaDisplayProps {
  formula: string;
  explanation?: string;
  variables?: Record<string, string>;
  title?: string;
}

export function FormulaDisplay({ 
  formula, 
  explanation, 
  variables,
  title = 'Formula'
}: FormulaDisplayProps) {
  const formulaRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (formulaRef.current && formula) {
      try {
        katex.render(formula, formulaRef.current, {
          displayMode: true,
          throwOnError: false,
          trust: true,
        });
      } catch (err) {
        console.error('KaTeX rendering error:', err);
      }
    }
  }, [formula]);
  
  return (
    <Card className="bg-slate-900 border-slate-700">
      <CardHeader>
        <CardTitle className="text-white text-lg">{title}</CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Formula */}
        <div 
          ref={formulaRef}
          className="text-white overflow-x-auto py-4 px-2 bg-slate-800 rounded-lg"
        />
        
        {/* Explanation */}
        {explanation && (
          <p className="text-slate-300 text-sm">
            {explanation}
          </p>
        )}
        
        {/* Variables */}
        {variables && Object.keys(variables).length > 0 && (
          <div className="space-y-2">
            <h4 className="text-slate-400 text-sm font-medium">Where:</h4>
            <ul className="space-y-1 text-sm">
              {Object.entries(variables).map(([symbol, description]) => (
                <li key={symbol} className="flex items-start gap-2 text-slate-300">
                  <span className="inline-flex items-center px-2 py-0.5 rounded bg-slate-800 font-mono text-purple-400">
                    {renderInlineFormula(symbol)}
                  </span>
                  <span className="flex-1">{description}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function renderInlineFormula(formula: string): React.ReactElement {
  const ref = useRef<HTMLSpanElement>(null);
  
  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(formula, ref.current, {
          displayMode: false,
          throwOnError: false,
        });
      } catch (err) {
        console.error('KaTeX inline rendering error:', err);
      }
    }
  }, [formula]);
  
  return <span ref={ref} />;
}

// Common formulas
export const FORMULAS = {
  attention: String.raw`\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V`,
  
  selfAttention: String.raw`\begin{aligned}
Q &= XW_Q \\
K &= XW_K \\
V &= XW_V
\end{aligned}`,
  
  multiHeadAttention: String.raw`\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O`,
  
  layerNorm: String.raw`\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta`,
  
  feedForward: String.raw`\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2`,
  
  positionalEncoding: String.raw`\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}`,
  
  moeGating: String.raw`\text{Router}(x) = \text{TopK}\left(\text{softmax}(xW_g)\right)`,
  
  moeOutput: String.raw`y = \sum_{i=1}^{k} g_i \cdot E_i(x)`,
  
  softmax: String.raw`\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}`,
  
  crossEntropy: String.raw`\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)`,
};

export const FORMULA_VARIABLES = {
  attention: {
    'Q': 'Query matrix',
    'K': 'Key matrix',
    'V': 'Value matrix',
    'd_k': 'Dimension of key vectors',
  },
  
  multiHeadAttention: {
    'h': 'Number of attention heads',
    'W^O': 'Output projection matrix',
  },
  
  layerNorm: {
    '\\mu': 'Mean of the input',
    '\\sigma^2': 'Variance of the input',
    '\\gamma': 'Learnable scale parameter',
    '\\beta': 'Learnable shift parameter',
    '\\epsilon': 'Small constant for numerical stability',
  },
  
  feedForward: {
    'W_1, b_1': 'First layer weights and bias',
    'W_2, b_2': 'Second layer weights and bias',
  },
  
  positionalEncoding: {
    'pos': 'Position in the sequence',
    'i': 'Dimension index',
    'd': 'Model dimension',
  },
  
  moe: {
    'W_g': 'Gating network weights',
    'g_i': 'Gate value for expert i',
    'E_i': 'Expert i function',
    'k': 'Number of active experts',
  },
};
