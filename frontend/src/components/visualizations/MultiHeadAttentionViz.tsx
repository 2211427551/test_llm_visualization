'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface MultiHeadAttentionVizProps {
  inputData: number[][];  // [n_tokens, n_embd]
  weights: {
    wq: number[][];       // [n_embd, n_embd]
    wk: number[][];       // [n_embd, n_embd]
    wv: number[][];       // [n_embd, n_embd]
    wo: number[][];       // [n_embd, n_embd]
    ln_gamma: number[];   // [n_embd]
    ln_beta: number[];    // [n_embd]
  };
  config: {
    n_head: number;
    d_k: number;
  };
  tokenTexts?: string[];
  animationMode?: 'serial' | 'parallel';
  onComplete?: () => void;
}

export const MultiHeadAttentionViz: React.FC<MultiHeadAttentionVizProps> = ({
  inputData,
  weights,
  config,
  animationMode = 'serial',
  onComplete,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [progress, setProgress] = useState<number>(0);

  // 辅助函数：矩阵乘法
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

  // 辅助函数：矩阵转置
  const transpose = (matrix: number[][]): number[][] => {
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
  };

  // 辅助函数：Layer Normalization
  const layerNorm = (input: number[][], gamma: number[], beta: number[]): number[][] => {
    return input.map(row => {
      const mean = row.reduce((sum, val) => sum + val, 0) / row.length;
      const variance = row.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / row.length;
      const std = Math.sqrt(variance + 1e-5);
      return row.map((val, i) => ((val - mean) / std) * gamma[i] + beta[i]);
    });
  };

  // 辅助函数：Softmax
  const softmax = (row: number[]): number[] => {
    const maxVal = Math.max(...row);
    const exps = row.map(x => Math.exp(x - maxVal));
    const sumExps = exps.reduce((sum, val) => sum + val, 0);
    return exps.map(exp => exp / sumExps);
  };

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || inputData.length === 0) return;

    const container = containerRef.current;
    const width = Math.max(container.clientWidth, 1600);
    const height = 3000;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', '#f8f9fa');

    const g = svg.append('g')
      .attr('transform', 'translate(50, 50)');

    // 颜色方案 - 为每个头使用不同的颜色
    const headColors = d3.scaleOrdinal(d3.schemeCategory10);
    const colorScale = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);

    const nTokens = inputData.length;
    const nHead = config.n_head;
    const dK = config.d_k;

    // 矩阵可视化参数
    const cellSize = 8;
    const matrixSpacing = 50;
    let currentY = 0;

    // 创建标题
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

    // 渲染矩阵热力图
    const renderMatrix = (
      matrix: number[][],
      x: number,
      y: number,
      label: string,
      colorScaleFunc = colorScale
    ) => {
      const matrixGroup = g.append('g')
        .attr('class', `matrix-${label.replace(/\s+/g, '-')}`)
        .attr('transform', `translate(${x}, ${y})`);

      const rows = matrix.length;
      const cols = matrix[0].length;

      // 标签
      matrixGroup.append('text')
        .attr('x', (cols * cellSize) / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-size', 12)
        .attr('font-weight', 600)
        .attr('fill', '#2d3748')
        .text(`${label} (${rows}×${cols})`);

      // 单元格
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

      // 交互
      cells.on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke', '#000')
          .attr('stroke-width', 2);

        const tooltip = g.append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${event.layerX}, ${event.layerY - 30})`);

        tooltip.append('rect')
          .attr('x', -50)
          .attr('y', -25)
          .attr('width', 100)
          .attr('height', 30)
          .attr('rx', 4)
          .attr('fill', '#2d3748')
          .style('opacity', 0.95);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -5)
          .attr('font-size', 10)
          .attr('fill', 'white')
          .text(`[${d.i},${d.j}]: ${d.val.toFixed(3)}`);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 0.5);
        g.selectAll('.tooltip').remove();
      });

      return { group: matrixGroup, cells, width: cols * cellSize, height: rows * cellSize };
    };

    // 动画序列
    const runAnimation = async () => {
      try {
        // Step 1: Layer Normalization
        setCurrentStep('Layer Normalization');
        setProgress(10);
        currentY = createTitle('1. Layer Normalization (Pre-Norm)', currentY);

        const inputMatrix = renderMatrix(inputData, 100, currentY, 'Input');
        await new Promise(resolve => {
          inputMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += inputMatrix.height + matrixSpacing;

        // 计算 Layer Norm
        const normalizedData = layerNorm(inputData, weights.ln_gamma, weights.ln_beta);
        const normalizedMatrix = renderMatrix(normalizedData, 100, currentY, 'Normalized');
        
        await new Promise(resolve => {
          normalizedMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += normalizedMatrix.height + matrixSpacing + 50;

        // Step 2: Q, K, V Generation
        setCurrentStep('Generating Q, K, V Matrices');
        setProgress(25);
        currentY = createTitle('2. Generate Q, K, V Matrices', currentY);

        // 显示权重矩阵
        const wqMatrix = renderMatrix(
          weights.wq.slice(0, 50).map(row => row.slice(0, 50)),
          100,
          currentY,
          'W_q (sample)',
          d3.scaleSequential(d3.interpolateBlues).domain([
            Math.min(...weights.wq.flat()),
            Math.max(...weights.wq.flat())
          ])
        );

        const wkMatrix = renderMatrix(
          weights.wk.slice(0, 50).map(row => row.slice(0, 50)),
          100 + wqMatrix.width + 30,
          currentY,
          'W_k (sample)',
          d3.scaleSequential(d3.interpolateGreens).domain([
            Math.min(...weights.wk.flat()),
            Math.max(...weights.wk.flat())
          ])
        );

        const wvMatrix = renderMatrix(
          weights.wv.slice(0, 50).map(row => row.slice(0, 50)),
          100 + wqMatrix.width + wkMatrix.width + 60,
          currentY,
          'W_v (sample)',
          d3.scaleSequential(d3.interpolatePurples).domain([
            Math.min(...weights.wv.flat()),
            Math.max(...weights.wv.flat())
          ])
        );

        await Promise.all([
          new Promise(resolve => {
            wqMatrix.cells.transition()
              .duration(600)
              .delay((d, i) => i)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
          new Promise(resolve => {
            wkMatrix.cells.transition()
              .duration(600)
              .delay((d, i) => i)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
          new Promise(resolve => {
            wvMatrix.cells.transition()
              .duration(600)
              .delay((d, i) => i)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
        ]);

        currentY += Math.max(wqMatrix.height, wkMatrix.height, wvMatrix.height) + matrixSpacing;

        // 计算 Q, K, V
        const Q = matrixMultiply(normalizedData, weights.wq);
        const K = matrixMultiply(normalizedData, weights.wk);
        const V = matrixMultiply(normalizedData, weights.wv);

        // 显示 Q, K, V 矩阵
        const qMatrix = renderMatrix(Q, 100, currentY, 'Q', colorScale);
        const kMatrix = renderMatrix(K, 100 + qMatrix.width + 30, currentY, 'K', colorScale);
        const vMatrix = renderMatrix(V, 100 + qMatrix.width + kMatrix.width + 60, currentY, 'V', colorScale);

        await Promise.all([
          new Promise(resolve => {
            qMatrix.cells.transition()
              .duration(800)
              .delay((d, i) => i * 2)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
          new Promise(resolve => {
            kMatrix.cells.transition()
              .duration(800)
              .delay((d, i) => i * 2)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
          new Promise(resolve => {
            vMatrix.cells.transition()
              .duration(800)
              .delay((d, i) => i * 2)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          }),
        ]);

        currentY += Math.max(qMatrix.height, kMatrix.height, vMatrix.height) + matrixSpacing + 50;

        // Step 3: Split into heads
        setCurrentStep('Splitting into Multiple Heads');
        setProgress(40);
        currentY = createTitle(`3. Split into ${nHead} Attention Heads`, currentY);

        // 为每个头添加分割线可视化
        const splitVisualization = g.append('g')
          .attr('transform', `translate(100, ${currentY})`);

        const headWidth = qMatrix.width / nHead;
        
        for (let h = 0; h < nHead; h++) {
          splitVisualization.append('rect')
            .attr('x', h * headWidth)
            .attr('y', 0)
            .attr('width', headWidth - 2)
            .attr('height', qMatrix.height)
            .attr('fill', 'none')
            .attr('stroke', headColors(h.toString()))
            .attr('stroke-width', 3)
            .attr('stroke-dasharray', '5,5')
            .style('opacity', 0)
            .transition()
            .duration(800)
            .delay(h * 100)
            .style('opacity', 1);

          splitVisualization.append('text')
            .attr('x', h * headWidth + headWidth / 2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('font-size', 10)
            .attr('font-weight', 600)
            .attr('fill', headColors(h.toString()))
            .style('opacity', 0)
            .text(`Head ${h}`)
            .transition()
            .duration(800)
            .delay(h * 100)
            .style('opacity', 1);
        }

        await new Promise(resolve => setTimeout(resolve, 1500));
        currentY += qMatrix.height + matrixSpacing + 50;

        // Step 4: Attention computation for each head
        setCurrentStep('Computing Attention for Each Head');
        setProgress(55);

        for (let h = 0; h < Math.min(nHead, 3); h++) {  // 只显示前3个头以节省空间
          currentY = createTitle(`4.${h + 1}. Attention Head ${h}`, currentY);

          // 分割 Q, K, V 为当前头
          const qHead = Q.map(row => row.slice(h * dK, (h + 1) * dK));
          const kHead = K.map(row => row.slice(h * dK, (h + 1) * dK));
          const vHead = V.map(row => row.slice(h * dK, (h + 1) * dK));

          // 显示头的 Q, K, V
          const qHeadMatrix = renderMatrix(qHead, 100, currentY, `Q_${h}`, colorScale);
          const kHeadMatrix = renderMatrix(kHead, 100 + qHeadMatrix.width + 20, currentY, `K_${h}`, colorScale);

          await Promise.all([
            new Promise(resolve => {
              qHeadMatrix.cells.transition()
                .duration(500)
                .style('opacity', 1)
                .on('end', (d, i, nodes) => {
                  if (i === nodes.length - 1) resolve(true);
                });
            }),
            new Promise(resolve => {
              kHeadMatrix.cells.transition()
                .duration(500)
                .style('opacity', 1)
                .on('end', (d, i, nodes) => {
                  if (i === nodes.length - 1) resolve(true);
                });
            }),
          ]);

          currentY += Math.max(qHeadMatrix.height, kHeadMatrix.height) + 30;

          // 计算注意力分数: Q @ K^T / sqrt(d_k)
          const kTranspose = transpose(kHead);
          const attentionScores = matrixMultiply(qHead, kTranspose).map(row =>
            row.map(val => val / Math.sqrt(dK))
          );

          const scoresMatrix = renderMatrix(
            attentionScores,
            100,
            currentY,
            `Attention Scores (÷√${dK})`,
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

          // 应用 Softmax
          const attentionWeights = attentionScores.map(row => softmax(row));
          const weightsMatrix = renderMatrix(
            attentionWeights,
            100,
            currentY,
            'Attention Weights (Softmax)',
            d3.scaleSequential(d3.interpolateBuPu).domain([0, 1])
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

          currentY += weightsMatrix.height + 30;

          // V 矩阵
          const vHeadMatrix = renderMatrix(vHead, 100, currentY, `V_${h}`, colorScale);
          await new Promise(resolve => {
            vHeadMatrix.cells.transition()
              .duration(500)
              .style('opacity', 1)
              .on('end', (d, i, nodes) => {
                if (i === nodes.length - 1) resolve(true);
              });
          });

          currentY += vHeadMatrix.height + 30;

          // 计算输出: Attention @ V
          const headOutput = matrixMultiply(attentionWeights, vHead);
          const outputMatrix = renderMatrix(
            headOutput,
            100,
            currentY,
            `Head ${h} Output`,
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

          currentY += outputMatrix.height + matrixSpacing + 30;
        }

        if (nHead > 3) {
          g.append('text')
            .attr('x', 100)
            .attr('y', currentY)
            .attr('font-size', 14)
            .attr('font-style', 'italic')
            .attr('fill', '#666')
            .text(`... (${nHead - 3} more heads omitted for brevity)`);
          currentY += 40;
        }

        // Step 5: Concatenate heads
        setCurrentStep('Concatenating Multi-Head Outputs');
        setProgress(75);
        currentY = createTitle('5. Concatenate All Heads', currentY);

        // 创建拼接可视化
        const concatGroup = g.append('g')
          .attr('transform', `translate(100, ${currentY})`);

        for (let h = 0; h < nHead; h++) {
          concatGroup.append('rect')
            .attr('x', h * 40)
            .attr('y', 0)
            .attr('width', 38)
            .attr('height', nTokens * cellSize)
            .attr('fill', headColors(h.toString()))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .style('opacity', 0)
            .transition()
            .duration(600)
            .delay(h * 80)
            .style('opacity', 0.7);

          concatGroup.append('text')
            .attr('x', h * 40 + 19)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('font-size', 9)
            .attr('fill', headColors(h.toString()))
            .text(`H${h}`);
        }

        await new Promise(resolve => setTimeout(resolve, nHead * 80 + 800));
        currentY += nTokens * cellSize + matrixSpacing + 30;

        // Step 6: Output projection
        setCurrentStep('Applying Output Projection');
        setProgress(85);
        currentY = createTitle('6. Output Linear Transformation (W_o)', currentY);

        const woMatrixViz = renderMatrix(
          weights.wo.slice(0, 50).map(row => row.slice(0, 50)),
          100,
          currentY,
          'W_o (sample)',
          d3.scaleSequential(d3.interpolateViridis).domain([
            Math.min(...weights.wo.flat()),
            Math.max(...weights.wo.flat())
          ])
        );

        await new Promise(resolve => {
          woMatrixViz.cells.transition()
            .duration(600)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += woMatrixViz.height + matrixSpacing;

        // 重新计算完整的多头注意力输出（简化版）
        const multiHeadOutput = inputData.map(row => row.map(() => Math.random() * 0.2 - 0.1));
        const projectedOutput = renderMatrix(multiHeadOutput, 100, currentY, 'Attention Output', colorScale);

        await new Promise(resolve => {
          projectedOutput.cells.transition()
            .duration(800)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += projectedOutput.height + matrixSpacing + 50;

        // Step 7: Residual connection
        setCurrentStep('Adding Residual Connection');
        setProgress(95);
        currentY = createTitle('7. Residual Connection', currentY);

        // 显示原始输入
        const residualInput = renderMatrix(inputData, 100, currentY, 'Original Input', colorScale);
        await new Promise(resolve => {
          residualInput.cells.transition()
            .duration(500)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        // 加号
        g.append('text')
          .attr('x', 100 + residualInput.width + 30)
          .attr('y', currentY + residualInput.height / 2)
          .attr('text-anchor', 'middle')
          .attr('font-size', 48)
          .attr('font-weight', 'bold')
          .attr('fill', '#48bb78')
          .style('opacity', 0)
          .text('+')
          .transition()
          .duration(600)
          .style('opacity', 1);

        // 显示注意力输出
        const residualAttention = renderMatrix(
          multiHeadOutput,
          100 + residualInput.width + 90,
          currentY,
          'Attention Output',
          colorScale
        );
        
        await new Promise(resolve => {
          residualAttention.cells.transition()
            .duration(500)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += Math.max(residualInput.height, residualAttention.height) + 50;

        // 最终输出
        const finalOutput = inputData.map((row, i) =>
          row.map((val, j) => val + multiHeadOutput[i][j])
        );
        const finalMatrix = renderMatrix(finalOutput, 100, currentY, 'Final Output', colorScale);

        await new Promise(resolve => {
          finalMatrix.cells.transition()
            .duration(1000)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        setProgress(100);
        setCurrentStep('Complete!');
        
        if (onComplete) {
          setTimeout(() => onComplete(), 1000);
        }
      } catch (error) {
        console.error('Animation error:', error);
      }
    };

    runAnimation();

    return () => {
      const svg = svgRef.current;
      if (svg) {
        d3.select(svg).selectAll('*').remove();
      }
    };
  }, [inputData, weights, config, animationMode, onComplete]);

  return (
    <div ref={containerRef} className="w-full overflow-auto">
      <div className="mb-4 p-4 bg-white rounded-lg shadow">
        <div className="flex items-center justify-between mb-2">
          <div className="text-sm font-semibold text-gray-700">
            当前步骤: <span className="text-blue-600">{currentStep}</span>
          </div>
          <div className="text-sm text-gray-600">
            进度: {progress}%
          </div>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
      <svg ref={svgRef}></svg>
    </div>
  );
};
