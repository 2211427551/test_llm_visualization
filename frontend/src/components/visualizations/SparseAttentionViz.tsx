'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface SparseAttentionVizProps {
  inputData: number[][];
  weights: {
    wq: number[][];
    wk: number[][];
    wv: number[][];
    wo: number[][];
    ln_gamma: number[];
    ln_beta: number[];
  };
  config: {
    n_head: number;
    d_k: number;
    sparse_pattern?: string;
    window_size?: number;
    block_size?: number;
    global_tokens?: number[];
  };
  attentionMask?: number[][];
  tokenTexts?: string[];
  showComparison?: boolean;
  onComplete?: () => void;
}

export const SparseAttentionViz: React.FC<SparseAttentionVizProps> = ({
  inputData,
  weights,
  config,
  attentionMask,
  tokenTexts = [],
  showComparison = false,
  onComplete,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [progress, setProgress] = useState<number>(0);
  const [sparsity, setSparsity] = useState<number>(0);

  // Helper functions
  const matrixMultiply = (a: number[][], b: number[][]): number[][] => {
    const m = a.length;
    const n = b[0].length;
    const p = b.length;
    const result: number[][] = Array(m).fill(0).map(() => Array(n).fill(0));
    
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < p; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
    return result;
  };

  const transpose = (matrix: number[][]): number[][] => {
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
  };

  const layerNorm = (input: number[][], gamma: number[], beta: number[]): number[][] => {
    return input.map(row => {
      const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
      const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;
      const std = Math.sqrt(variance + 1e-5);
      return row.map((val, i) => ((val - mean) / std) * gamma[i] + beta[i]);
    });
  };

  const softmax = (row: number[]): number[] => {
    const maxVal = Math.max(...row);
    const exps = row.map(x => Math.exp(x - maxVal));
    const sumExps = exps.reduce((sum, val) => sum + val, 0);
    return exps.map(exp => exp / sumExps);
  };

  const generateSparseMask = (nTokens: number): number[][] => {
    if (attentionMask) {
      return attentionMask;
    }

    const mask = Array(nTokens).fill(0).map(() => Array(nTokens).fill(0));
    const pattern = config.sparse_pattern || 'sliding_window';
    const windowSize = config.window_size || 3;
    const blockSize = config.block_size || 4;
    const globalTokens = config.global_tokens || [];

    switch (pattern) {
      case 'sliding_window':
        for (let i = 0; i < nTokens; i++) {
          const start = Math.max(0, i - windowSize);
          const end = Math.min(nTokens, i + windowSize + 1);
          for (let j = start; j < end; j++) {
            mask[i][j] = 1;
          }
        }
        break;

      case 'global_local':
        for (let i = 0; i < nTokens; i++) {
          const start = Math.max(0, i - windowSize);
          const end = Math.min(nTokens, i + windowSize + 1);
          for (let j = start; j < end; j++) {
            mask[i][j] = 1;
          }
        }
        globalTokens.forEach(idx => {
          if (idx >= 0 && idx < nTokens) {
            for (let j = 0; j < nTokens; j++) {
              mask[idx][j] = 1;
              mask[j][idx] = 1;
            }
          }
        });
        break;

      case 'blocked':
        const nBlocks = Math.ceil(nTokens / blockSize);
        for (let i = 0; i < nTokens; i++) {
          const blockI = Math.floor(i / blockSize);
          for (let blockJ = Math.max(0, blockI - 1); blockJ <= Math.min(nBlocks - 1, blockI + 1); blockJ++) {
            const start = blockJ * blockSize;
            const end = Math.min((blockJ + 1) * blockSize, nTokens);
            for (let j = start; j < end; j++) {
              mask[i][j] = 1;
            }
          }
        }
        break;

      case 'dense':
      default:
        for (let i = 0; i < nTokens; i++) {
          for (let j = 0; j < nTokens; j++) {
            mask[i][j] = 1;
          }
        }
    }

    return mask;
  };

  const calculateSparsity = (mask: number[][]): number => {
    const total = mask.length * mask[0].length;
    const masked = mask.flat().filter(v => v === 0).length;
    return masked / total;
  };

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || inputData.length === 0) return;

    const container = containerRef.current;
    const width = Math.max(container.clientWidth, 1600);
    const height = showComparison ? 4000 : 3500;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', '#f8f9fa');

    const g = svg.append('g')
      .attr('transform', 'translate(50, 50)');

    const headColors = d3.scaleOrdinal(d3.schemeCategory10);
    const colorScale = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);

    const nTokens = inputData.length;
    const nHead = config.n_head;
    const dK = config.d_k;

    const cellSize = 8;
    const matrixSpacing = 50;
    let currentY = 0;

    const mask = generateSparseMask(nTokens);
    const sparsityValue = calculateSparsity(mask);
    setSparsity(sparsityValue);

    const createTitle = (text: string, y: number) => {
      g.append('text')
        .attr('x', width / 2 - 50)
        .attr('y', y)
        .attr('text-anchor', 'middle')
        .attr('font-size', 18)
        .attr('font-weight', 'bold')
        .attr('fill', '#2d3748')
        .text(text);
      return y + 30;
    };

    const renderMatrix = (
      matrix: number[][],
      x: number,
      y: number,
      label: string,
      colorScaleFunc = colorScale,
      showMask = false,
      maskData: number[][] | null = null
    ) => {
      const matrixGroup = g.append('g')
        .attr('class', `matrix-${label.replace(/\s+/g, '-')}`)
        .attr('transform', `translate(${x}, ${y})`);

      const rows = matrix.length;
      const cols = matrix[0].length;

      matrixGroup.append('text')
        .attr('x', (cols * cellSize) / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .attr('fill', '#2d3748')
        .text(`${label} (${rows}×${cols})`);

      const cells = matrixGroup.selectAll('.cell')
        .data(matrix.flatMap((row, i) => row.map((val, j) => ({ val, i, j }))))
        .join('rect')
        .attr('class', 'cell')
        .attr('x', d => d.j * cellSize)
        .attr('y', d => d.i * cellSize)
        .attr('width', cellSize - 0.5)
        .attr('height', cellSize - 0.5)
        .attr('fill', d => colorScaleFunc(d.val))
        .attr('stroke', '#fff')
        .attr('stroke-width', 0.5)
        .style('opacity', 0);

      if (showMask && maskData) {
        const maskCells = matrixGroup.selectAll('.mask-cell')
          .data(maskData.flatMap((row, i) => row.map((val, j) => ({ val, i, j }))))
          .join('rect')
          .attr('class', 'mask-cell')
          .attr('x', d => d.j * cellSize)
          .attr('y', d => d.i * cellSize)
          .attr('width', cellSize - 0.5)
          .attr('height', cellSize - 0.5)
          .attr('fill', d => d.val === 0 ? '#000' : 'none')
          .attr('stroke', 'none')
          .style('opacity', 0)
          .attr('pointer-events', 'none');

        return { group: matrixGroup, cells, maskCells, width: cols * cellSize, height: rows * cellSize };
      }

      cells.on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke', '#000')
          .attr('stroke-width', 2);

        const isMasked = maskData && maskData[d.i][d.j] === 0;
        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${event.layerX}, ${event.layerY - 30})`);

        tooltip.append('rect')
          .attr('x', -60)
          .attr('y', -25)
          .attr('width', 120)
          .attr('height', 30)
          .attr('rx', 4)
          .attr('fill', '#2d3748')
          .style('opacity', 0.95);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -5)
          .attr('font-size', 10)
          .attr('fill', 'white')
          .text(isMasked ? `[${d.i},${d.j}]: Masked` : `[${d.i},${d.j}]: ${d.val.toFixed(3)}`);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 0.5);
        g.selectAll('.tooltip').remove();
      });

      return { group: matrixGroup, cells, width: cols * cellSize, height: rows * cellSize };
    };

    const runAnimation = async () => {
      try {
        // Sparse pattern info
        setCurrentStep('Sparse Attention Pattern');
        setProgress(5);
        currentY = createTitle(
          `Sparse Attention: ${config.sparse_pattern || 'sliding_window'} (Sparsity: ${(sparsityValue * 100).toFixed(1)}%)`,
          currentY
        );

        // Show sparsity statistics
        const statsGroup = g.append('g')
          .attr('transform', `translate(100, ${currentY})`);

        statsGroup.append('text')
          .attr('x', 0)
          .attr('y', 0)
          .attr('font-size', 14)
          .attr('fill', '#2d3748')
          .text(`计算复杂度: O(n²) → O(n·${config.window_size || 'w'})`);

        statsGroup.append('text')
          .attr('x', 0)
          .attr('y', 25)
          .attr('font-size', 14)
          .attr('fill', '#2d3748')
          .text(`屏蔽连接: ${(sparsityValue * 100).toFixed(1)}%`);

        currentY += 60;

        // Step 1: Show sparse mask pattern
        setCurrentStep('Sparse Mask Pattern');
        setProgress(15);
        currentY = createTitle('1. Sparse Attention Mask', currentY);

        const maskMatrix = renderMatrix(
          mask,
          100,
          currentY,
          'Attention Mask',
          d3.scaleSequential(d3.interpolateGreys).domain([0, 1])
        );

        await new Promise(resolve => {
          maskMatrix.cells.transition()
            .duration(1000)
            .delay((d, i) => Math.floor(i / nTokens) * 50)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += maskMatrix.height + matrixSpacing + 50;

        // Step 2: Layer Normalization
        setCurrentStep('Layer Normalization');
        setProgress(25);
        currentY = createTitle('2. Layer Normalization', currentY);

        const normalizedData = layerNorm(inputData, weights.ln_gamma, weights.ln_beta);
        const normalizedMatrix = renderMatrix(normalizedData, 100, currentY, 'Normalized Input');
        
        await new Promise(resolve => {
          normalizedMatrix.cells.transition()
            .duration(600)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += normalizedMatrix.height + matrixSpacing + 50;

        // Step 3: Q, K, V Generation
        setCurrentStep('Generating Q, K, V');
        setProgress(40);
        currentY = createTitle('3. Generate Q, K, V Matrices', currentY);

        const Q = matrixMultiply(normalizedData, weights.wq);
        const K = matrixMultiply(normalizedData, weights.wk);
        const V = matrixMultiply(normalizedData, weights.wv);

        const qMatrix = renderMatrix(Q, 100, currentY, 'Q', colorScale);
        const kMatrix = renderMatrix(K, 100 + qMatrix.width + 30, currentY, 'K', colorScale);
        const vMatrix = renderMatrix(V, 100 + qMatrix.width + kMatrix.width + 60, currentY, 'V', colorScale);

        await Promise.all([
          new Promise(resolve => {
            qMatrix.cells.transition()
              .duration(600)
              .delay((d, i) => i * 2)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
          new Promise(resolve => {
            kMatrix.cells.transition()
              .duration(600)
              .delay((d, i) => i * 2)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
          new Promise(resolve => {
            vMatrix.cells.transition()
              .duration(600)
              .delay((d, i) => i * 2)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
        ]);

        currentY += Math.max(qMatrix.height, kMatrix.height, vMatrix.height) + matrixSpacing + 50;

        // Step 4: Attention computation with sparse mask (first head only)
        setCurrentStep('Computing Sparse Attention');
        setProgress(60);
        currentY = createTitle('4. Sparse Attention Computation (Head 0)', currentY);

        const qHead = Q.map(row => row.slice(0, dK));
        const kHead = K.map(row => row.slice(0, dK));
        const vHead = V.map(row => row.slice(0, dK));

        // Calculate attention scores
        const kTranspose = transpose(kHead);
        const attentionScores = matrixMultiply(qHead, kTranspose).map(row =>
          row.map(val => val / Math.sqrt(dK))
        );

        // Show scores before masking
        const scoresMatrix = renderMatrix(
          attentionScores,
          100,
          currentY,
          'Attention Scores (Before Mask)',
          d3.scaleSequential(d3.interpolateYlOrRd).domain([
            Math.min(...attentionScores.flat()),
            Math.max(...attentionScores.flat())
          ])
        );

        await new Promise(resolve => {
          scoresMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i * 3)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += scoresMatrix.height + 30;

        // Apply sparse mask
        setCurrentStep('Applying Sparse Mask');
        setProgress(70);
        currentY = createTitle('5. Apply Sparse Mask', currentY);

        const maskedScores = attentionScores.map((row, i) =>
          row.map((val, j) => mask[i][j] === 0 ? -1e9 : val)
        );

        const maskedScoresMatrix = renderMatrix(
          maskedScores,
          100,
          currentY,
          'Masked Scores',
          d3.scaleSequential(d3.interpolateYlOrRd).domain([
            Math.min(...maskedScores.flat().filter((v: number) => v > -1e8)),
            Math.max(...maskedScores.flat())
          ]),
          true,
          mask
        );

        await new Promise(resolve => {
          maskedScoresMatrix.cells.transition()
            .duration(600)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        if (maskedScoresMatrix.maskCells) {
          await new Promise(resolve => {
            maskedScoresMatrix.maskCells.transition()
              .duration(800)
              .delay((d, i) => Math.floor(i / nTokens) * 100)
              .style('opacity', 0.7)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          });
        }

        currentY += maskedScoresMatrix.height + 30;

        // Apply Softmax
        setCurrentStep('Softmax (Sparse)');
        setProgress(80);
        currentY = createTitle('6. Softmax (Sparse Normalized)', currentY);

        const attentionWeights = maskedScores.map(row => softmax(row));
        const weightsMatrix = renderMatrix(
          attentionWeights,
          100,
          currentY,
          'Sparse Attention Weights',
          d3.scaleSequential(d3.interpolateBuPu).domain([0, 1]),
          true,
          mask
        );

        await new Promise(resolve => {
          weightsMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => Math.floor(i / nTokens) * 100)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        if (weightsMatrix.maskCells) {
          await new Promise(resolve => {
            weightsMatrix.maskCells.transition()
              .duration(500)
              .style('opacity', 0.7)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          });
        }

        currentY += weightsMatrix.height + 30;

        // Multiply with V
        setCurrentStep('Weighted Sum with V');
        setProgress(90);
        currentY = createTitle('7. Attention @ V', currentY);

        const headOutput = matrixMultiply(attentionWeights, vHead);
        const outputMatrix = renderMatrix(
          headOutput,
          100,
          currentY,
          'Sparse Attention Output',
          colorScale
        );

        await new Promise(resolve => {
          outputMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += outputMatrix.height + matrixSpacing;

        // Add summary
        setCurrentStep('Complete');
        setProgress(100);

        const summaryGroup = g.append('g')
          .attr('transform', `translate(100, ${currentY})`);

        summaryGroup.append('text')
          .attr('x', 0)
          .attr('y', 0)
          .attr('font-size', 16)
          .attr('font-weight', 'bold')
          .attr('fill', '#2d3748')
          .text('稀疏注意力优势:');

        summaryGroup.append('text')
          .attr('x', 0)
          .attr('y', 30)
          .attr('font-size', 14)
          .attr('fill', '#4a5568')
          .text(`✓ 计算量减少 ${(sparsityValue * 100).toFixed(1)}%`);

        summaryGroup.append('text')
          .attr('x', 0)
          .attr('y', 55)
          .attr('font-size', 14)
          .attr('fill', '#4a5568')
          .text(`✓ 内存占用降低`);

        summaryGroup.append('text')
          .attr('x', 0)
          .attr('y', 80)
          .attr('font-size', 14)
          .attr('fill', '#4a5568')
          .text(`✓ 支持更长序列`);

        if (onComplete) onComplete();
      } catch (error) {
        console.error('Animation error:', error);
      }
    };

    runAnimation();
  }, [inputData, weights, config, attentionMask, showComparison, onComplete]);

  return (
    <div ref={containerRef} className="w-full">
      <div className="mb-4 p-4 bg-white rounded-lg shadow">
        <div className="mb-2">
          <span className="font-semibold">当前步骤: </span>
          <span className="text-blue-600">{currentStep}</span>
        </div>
        <div className="mb-2">
          <span className="font-semibold">进度: </span>
          <span className="text-blue-600">{progress}%</span>
        </div>
        <div className="mb-2">
          <span className="font-semibold">稀疏度: </span>
          <span className="text-purple-600">{(sparsity * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
      <div className="overflow-auto max-h-[800px] border border-gray-300 rounded-lg">
        <svg ref={svgRef} />
      </div>
    </div>
  );
};
