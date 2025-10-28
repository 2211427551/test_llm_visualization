'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface MoEFFNVizProps {
  inputData: number[][];  // [n_tokens, n_embd]
  weights: {
    ln_gamma: number[];   // [n_embd]
    ln_beta: number[];    // [n_embd]
    gate_weights: number[][];  // [n_embd, n_experts]
    experts: {
      w1: number[][];  // [n_embd, d_ff]
      w2: number[][];  // [d_ff, n_embd]
    }[];
  };
  config: {
    n_experts: number;
    top_k: number;
    d_ff: number;
    n_embd: number;
  };
  tokenTexts?: string[];
  animationMode?: 'serial' | 'parallel';
  onComplete?: () => void;
}

export const MoEFFNViz: React.FC<MoEFFNVizProps> = ({
  inputData,
  weights,
  config,
  tokenTexts = [],
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

  // 辅助函数：向量乘法（用于单个向量与矩阵相乘）
  const vectorMatrixMultiply = (vec: number[], matrix: number[][]): number[] => {
    const result: number[] = Array(matrix[0].length).fill(0);
    for (let j = 0; j < matrix[0].length; j++) {
      for (let i = 0; i < vec.length; i++) {
        result[j] += vec[i] * matrix[i][j];
      }
    }
    return result;
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

  // 辅助函数：ReLU
  const relu = (x: number): number => Math.max(0, x);

  // 辅助函数：获取Top-K索引
  const getTopK = (arr: number[], k: number): number[] => {
    return arr
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val)
      .slice(0, k)
      .map(item => item.idx);
  };

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || inputData.length === 0) return;

    const container = containerRef.current;
    const width = Math.max(container.clientWidth, 1800);
    const height = 4500;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', '#f8f9fa');

    const g = svg.append('g')
      .attr('transform', 'translate(50, 50)');

    // 颜色方案
    const expertColors = d3.scaleOrdinal(d3.schemeCategory10);
    const colorScale = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);

    const nTokens = inputData.length;
    const nExperts = config.n_experts;
    const topK = config.top_k;
    const dFF = config.d_ff;

    // 矩阵可视化参数
    const cellSize = 6;
    const matrixSpacing = 50;
    let currentY = 0;

    // 创建标题
    const createTitle = (text: string, y: number, fontSize = 18) => {
      g.append('text')
        .attr('x', width / 2 - 50)
        .attr('y', y)
        .attr('text-anchor', 'middle')
        .attr('font-size', fontSize)
        .attr('font-weight', 'bold')
        .attr('fill', '#2d3748')
        .text(text);
      return y + 30;
    };

    // 创建说明文本
    const createDescription = (text: string, y: number) => {
      g.append('text')
        .attr('x', 100)
        .attr('y', y)
        .attr('font-size', 12)
        .attr('fill', '#4a5568')
        .text(text);
      return y + 20;
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
          .attr('transform', `translate(${x + d.j * cellSize}, ${y + d.i * cellSize - 30})`);

        tooltip.append('rect')
          .attr('x', -40)
          .attr('y', -25)
          .attr('width', 80)
          .attr('height', 30)
          .attr('rx', 4)
          .attr('fill', '#2d3748')
          .style('opacity', 0.95);

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('y', -5)
          .attr('font-size', 10)
          .attr('fill', 'white')
          .text(`${d.val.toFixed(3)}`);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 0.5);
        g.selectAll('.tooltip').remove();
      });

      return { group: matrixGroup, cells, width: cols * cellSize, height: rows * cellSize };
    };

    // 渲染向量
    const renderVector = (
      vector: number[],
      x: number,
      y: number,
      label: string,
      colorScaleFunc = colorScale,
      vertical = false
    ) => {
      const vectorGroup = g.append('g')
        .attr('class', `vector-${label.replace(/\s+/g, '-')}`)
        .attr('transform', `translate(${x}, ${y})`);

      vectorGroup.append('text')
        .attr('x', vertical ? -10 : vector.length * cellSize / 2)
        .attr('y', vertical ? vector.length * cellSize / 2 : -10)
        .attr('text-anchor', vertical ? 'end' : 'middle')
        .attr('font-size', 11)
        .attr('font-weight', 600)
        .attr('fill', '#2d3748')
        .text(`${label} (${vector.length})`);

      const cells = vectorGroup.selectAll('.cell')
        .data(vector.map((val, i) => ({ val, i })))
        .join('rect')
        .attr('class', 'cell')
        .attr('x', d => vertical ? 0 : d.i * cellSize)
        .attr('y', d => vertical ? d.i * cellSize : 0)
        .attr('width', vertical ? cellSize - 0.5 : cellSize - 0.5)
        .attr('height', vertical ? cellSize - 0.5 : cellSize - 0.5)
        .attr('fill', d => colorScaleFunc(d.val))
        .attr('stroke', '#fff')
        .attr('stroke-width', 0.5)
        .style('opacity', 0);

      cells.on('mouseover', function() {
        d3.select(this)
          .attr('stroke', '#000')
          .attr('stroke-width', 2);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', '#fff')
          .attr('stroke-width', 0.5);
      });

      return {
        group: vectorGroup,
        cells,
        width: vertical ? cellSize : vector.length * cellSize,
        height: vertical ? vector.length * cellSize : cellSize
      };
    };

    // 动画序列
    const runAnimation = async () => {
      try {
        // Step 0: 显示输入tokens
        setCurrentStep('Input Tokens');
        setProgress(5);
        currentY = createTitle('MoE Feed-Forward Network Visualization', currentY, 24);
        currentY = createDescription(
          `Routing ${nTokens} token(s) to ${nExperts} experts, selecting top-${topK} for each token`,
          currentY
        );
        currentY += 30;

        // Step 1: Layer Normalization (Pre-Norm)
        setCurrentStep('Layer Normalization (Pre-Norm for FFN)');
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

        // Step 2: 门控网络/路由器（Gating Network）
        setCurrentStep('Gating Network - Computing Expert Selection Logits');
        setProgress(20);
        currentY = createTitle('2. Gating Network / Router', currentY);
        currentY = createDescription(
          'Each token multiplies with gate weights to produce expert logits',
          currentY
        );

        // 显示门控权重矩阵
        const gateMatrix = renderMatrix(
          weights.gate_weights,
          100,
          currentY,
          'Gate Weights W_gate',
          d3.scaleSequential(d3.interpolateViridis).domain([
            Math.min(...weights.gate_weights.flat()),
            Math.max(...weights.gate_weights.flat())
          ])
        );

        await new Promise(resolve => {
          gateMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += gateMatrix.height + matrixSpacing;

        // 计算门控logits
        const gateLogits = matrixMultiply(normalizedData, weights.gate_weights);
        const logitsMatrix = renderMatrix(
          gateLogits,
          100,
          currentY,
          'Gate Logits (before softmax)',
          d3.scaleSequential(d3.interpolateRdYlGn).domain([
            Math.min(...gateLogits.flat()),
            Math.max(...gateLogits.flat())
          ])
        );

        await new Promise(resolve => {
          logitsMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += logitsMatrix.height + matrixSpacing + 50;

        // Step 3: Softmax归一化 - 专家选择概率
        setCurrentStep('Softmax - Expert Selection Probabilities');
        setProgress(30);
        currentY = createTitle('3. Expert Selection Probabilities (Softmax)', currentY);

        // 计算专家选择概率
        const expertProbs = gateLogits.map(row => softmax(row));
        const probsMatrix = renderMatrix(
          expertProbs,
          100,
          currentY,
          'Expert Probabilities',
          d3.scaleSequential(d3.interpolateBuPu).domain([0, 1])
        );

        await new Promise(resolve => {
          probsMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += probsMatrix.height + matrixSpacing + 50;

        // Step 4: 条形图展示专家概率
        setCurrentStep('Expert Selection Bar Charts');
        setProgress(40);
        currentY = createTitle('4. Expert Selection Visualization', currentY);

        // 为每个token创建条形图
        const barChartHeight = 120;
        const barChartWidth = Math.min(600, (width - 200) / Math.max(1, nTokens));
        const barWidth = barChartWidth / nExperts - 2;

        for (let t = 0; t < nTokens; t++) {
          const tokenProbs = expertProbs[t];
          const topKIndices = getTopK(tokenProbs, topK);

          const barX = 100 + t * (barChartWidth + 50);
          const barY = currentY;

          const barGroup = g.append('g')
            .attr('transform', `translate(${barX}, ${barY})`);

          barGroup.append('text')
            .attr('x', barChartWidth / 2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('font-size', 12)
            .attr('font-weight', 600)
            .attr('fill', '#2d3748')
            .text(`Token ${t}${tokenTexts[t] ? ` "${tokenTexts[t]}"` : ''}`);

          const yScale = d3.scaleLinear()
            .domain([0, Math.max(...tokenProbs)])
            .range([0, barChartHeight]);

          const bars = barGroup.selectAll('.bar')
            .data(tokenProbs.map((val, idx) => ({ val, idx, isTopK: topKIndices.includes(idx) })))
            .join('rect')
            .attr('class', 'bar')
            .attr('x', d => d.idx * (barChartWidth / nExperts))
            .attr('y', barChartHeight)
            .attr('width', barWidth)
            .attr('height', 0)
            .attr('fill', d => d.isTopK ? expertColors(d.idx.toString()) : '#d0d0d0')
            .attr('stroke', d => d.isTopK ? '#000' : 'none')
            .attr('stroke-width', d => d.isTopK ? 2 : 0)
            .style('opacity', 0.8);

          bars.transition()
            .duration(800)
            .delay((d, i) => i * 30)
            .attr('y', d => barChartHeight - yScale(d.val))
            .attr('height', d => yScale(d.val));

          // 添加X轴标签
          barGroup.selectAll('.x-label')
            .data(tokenProbs.map((_, idx) => idx))
            .join('text')
            .attr('class', 'x-label')
            .attr('x', d => d * (barChartWidth / nExperts) + barWidth / 2)
            .attr('y', barChartHeight + 15)
            .attr('text-anchor', 'middle')
            .attr('font-size', 9)
            .attr('fill', '#4a5568')
            .text(d => `E${d}`)
            .style('opacity', 0)
            .transition()
            .duration(500)
            .delay(800)
            .style('opacity', 1);

          // 添加概率值标签（仅top-k）
          bars.filter(d => d.isTopK)
            .each(function(d) {
              barGroup.append('text')
                .attr('x', d.idx * (barChartWidth / nExperts) + barWidth / 2)
                .attr('y', barChartHeight - yScale(d.val) - 5)
                .attr('text-anchor', 'middle')
                .attr('font-size', 9)
                .attr('font-weight', 600)
                .attr('fill', '#000')
                .style('opacity', 0)
                .text(d.val.toFixed(3))
                .transition()
                .duration(500)
                .delay(1000)
                .style('opacity', 1);
            });

          // 交互
          bars.on('mouseover', function() {
            d3.select(this)
              .attr('stroke', '#000')
              .attr('stroke-width', 3)
              .style('opacity', 1);
          })
          .on('mouseout', function() {
            const bar = d3.select(this);
            const barData = bar.datum() as { val: number; idx: number; isTopK: boolean };
            bar.attr('stroke', barData.isTopK ? '#000' : 'none')
              .attr('stroke-width', barData.isTopK ? 2 : 0)
              .style('opacity', 0.8);
          });
        }

        await new Promise(resolve => setTimeout(resolve, 1500));
        currentY += barChartHeight + matrixSpacing + 80;

        // Step 5: 专家网络布局
        setCurrentStep('Expert Networks Layout');
        setProgress(50);
        currentY = createTitle('5. Expert Networks (Parallel FFNs)', currentY);
        currentY = createDescription(
          `${nExperts} independent expert networks, each is a 2-layer FFN`,
          currentY
        );

        // 计算每个专家的布局
        const expertsPerRow = Math.min(4, nExperts);
        const expertWidth = 180;
        const expertHeight = 200;
        const expertSpacing = 40;
        const expertStartX = 100;
        const expertStartY = currentY;

        // 绘制所有专家网络
        const expertGroups: d3.Selection<SVGGElement, unknown, null, undefined>[] = [];
        for (let e = 0; e < nExperts; e++) {
          const row = Math.floor(e / expertsPerRow);
          const col = e % expertsPerRow;
          const expertX = expertStartX + col * (expertWidth + expertSpacing);
          const expertY = expertStartY + row * (expertHeight + expertSpacing);

          const expertGroup = g.append('g')
            .attr('class', `expert-${e}`)
            .attr('transform', `translate(${expertX}, ${expertY})`);

          expertGroups.push(expertGroup);

          // 专家边框
          expertGroup.append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', expertWidth)
            .attr('height', expertHeight)
            .attr('rx', 8)
            .attr('fill', 'white')
            .attr('stroke', expertColors(e.toString()))
            .attr('stroke-width', 2)
            .style('opacity', 0.3);

          // 专家标题
          expertGroup.append('text')
            .attr('x', expertWidth / 2)
            .attr('y', 25)
            .attr('text-anchor', 'middle')
            .attr('font-size', 14)
            .attr('font-weight', 'bold')
            .attr('fill', expertColors(e.toString()))
            .text(`Expert ${e}`);

          // 专家结构示意
          expertGroup.append('text')
            .attr('x', expertWidth / 2)
            .attr('y', 50)
            .attr('text-anchor', 'middle')
            .attr('font-size', 10)
            .attr('fill', '#4a5568')
            .text(`Linear(${config.n_embd} → ${dFF})`);

          expertGroup.append('text')
            .attr('x', expertWidth / 2)
            .attr('y', 70)
            .attr('text-anchor', 'middle')
            .attr('font-size', 10)
            .attr('fill', '#4a5568')
            .text('↓ ReLU');

          expertGroup.append('text')
            .attr('x', expertWidth / 2)
            .attr('y', 90)
            .attr('text-anchor', 'middle')
            .attr('font-size', 10)
            .attr('fill', '#4a5568')
            .text(`Linear(${dFF} → ${config.n_embd})`);

          // 权重矩阵示意（缩略）
          const w1Sample = weights.experts[e].w1.slice(0, 10).map(row => row.slice(0, 20));
          const w1Viz = expertGroup.append('g')
            .attr('transform', `translate(${expertWidth / 2 - 30}, 110)`);
          
          w1Viz.selectAll('.w1-cell')
            .data(w1Sample.flatMap((row, i) => row.map((val, j) => ({ val, i, j }))))
            .join('rect')
            .attr('x', d => d.j * 3)
            .attr('y', d => d.i * 3)
            .attr('width', 2.5)
            .attr('height', 2.5)
            .attr('fill', d => colorScale(d.val))
            .style('opacity', 0);

          w1Viz.selectAll('.w1-cell')
            .transition()
            .duration(500)
            .delay(e * 50)
            .style('opacity', 1);

          const w2Sample = weights.experts[e].w2.slice(0, 20).map(row => row.slice(0, 10));
          const w2Viz = expertGroup.append('g')
            .attr('transform', `translate(${expertWidth / 2 - 30}, 155)`);
          
          w2Viz.selectAll('.w2-cell')
            .data(w2Sample.flatMap((row, i) => row.map((val, j) => ({ val, i, j }))))
            .join('rect')
            .attr('x', d => d.j * 3)
            .attr('y', d => d.i * 3)
            .attr('width', 2.5)
            .attr('height', 2.5)
            .attr('fill', d => colorScale(d.val))
            .style('opacity', 0);

          w2Viz.selectAll('.w2-cell')
            .transition()
            .duration(500)
            .delay(e * 50 + 250)
            .style('opacity', 1);
        }

        await new Promise(resolve => setTimeout(resolve, 1500));
        currentY = expertStartY + Math.ceil(nExperts / expertsPerRow) * (expertHeight + expertSpacing) + 50;

        // Step 6: Token到专家的路由动画
        setCurrentStep('Routing Tokens to Selected Experts');
        setProgress(60);
        currentY = createTitle('6. Routing Animation', currentY);

        // 创建token表示
        const tokenY = currentY;
        const tokenSpacing = Math.min(100, (width - 200) / nTokens);
        const tokenStartX = (width - tokenSpacing * nTokens) / 2;

        for (let t = 0; t < nTokens; t++) {
          const tokenX = tokenStartX + t * tokenSpacing;
          const tokenGroup = g.append('g')
            .attr('class', `token-routing-${t}`)
            .attr('transform', `translate(${tokenX}, ${tokenY})`);

          tokenGroup.append('circle')
            .attr('r', 20)
            .attr('fill', d3.interpolateBlues(0.6))
            .attr('stroke', '#2c5282')
            .attr('stroke-width', 2);

          tokenGroup.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', 5)
            .attr('font-size', 12)
            .attr('font-weight', 'bold')
            .attr('fill', 'white')
            .text(`T${t}`);
        }

        await new Promise(resolve => setTimeout(resolve, 500));

        // 绘制路由线
        const routingLines = g.append('g').attr('class', 'routing-lines');

        for (let t = 0; t < nTokens; t++) {
          const topKIndices = getTopK(expertProbs[t], topK);
          const tokenX = tokenStartX + t * tokenSpacing;
          const tokenYPos = tokenY;

          for (const expertIdx of topKIndices) {
            const row = Math.floor(expertIdx / expertsPerRow);
            const col = expertIdx % expertsPerRow;
            const expertX = expertStartX + col * (expertWidth + expertSpacing) + expertWidth / 2;
            const expertYPos = expertStartY + row * (expertHeight + expertSpacing) + expertHeight / 2;

            const gateWeight = expertProbs[t][expertIdx];

            // 创建曲线路径
            const pathData = [
              { x: tokenX, y: tokenYPos },
              { x: tokenX, y: tokenYPos + (expertYPos - tokenYPos - 200) / 2 },
              { x: expertX, y: expertYPos - (expertYPos - tokenYPos - 200) / 2 },
              { x: expertX, y: expertYPos }
            ];

            const lineGenerator = d3.line<{x: number, y: number}>()
              .x(d => d.x)
              .y(d => d.y)
              .curve(d3.curveBasis);

            const path = routingLines.append('path')
              .attr('d', lineGenerator(pathData))
              .attr('fill', 'none')
              .attr('stroke', expertColors(expertIdx.toString()))
              .attr('stroke-width', gateWeight * 8 + 1)
              .attr('opacity', 0.6)
              .attr('marker-end', `url(#arrow-${expertIdx})`)
              .style('opacity', 0);

            // 定义箭头标记
            svg.append('defs')
              .append('marker')
              .attr('id', `arrow-${expertIdx}`)
              .attr('viewBox', '0 -5 10 10')
              .attr('refX', 8)
              .attr('refY', 0)
              .attr('markerWidth', 6)
              .attr('markerHeight', 6)
              .attr('orient', 'auto')
              .append('path')
              .attr('d', 'M0,-5L10,0L0,5')
              .attr('fill', expertColors(expertIdx.toString()));

            // 动画显示路由线
            const pathLength = (path.node() as SVGPathElement).getTotalLength();
            path.attr('stroke-dasharray', `${pathLength} ${pathLength}`)
              .attr('stroke-dashoffset', pathLength)
              .style('opacity', 0.6)
              .transition()
              .duration(1000)
              .delay(t * 200 + topKIndices.indexOf(expertIdx) * 100)
              .attr('stroke-dashoffset', 0);

            // 高亮选中的专家
            expertGroups[expertIdx].select('rect')
              .transition()
              .duration(500)
              .delay(t * 200 + topKIndices.indexOf(expertIdx) * 100)
              .style('opacity', 1)
              .attr('stroke-width', 4)
              .style('filter', `drop-shadow(0 0 10px ${expertColors(expertIdx.toString())})`);
          }
        }

        await new Promise(resolve => setTimeout(resolve, nTokens * 200 + topK * 100 + 1000));
        currentY = tokenY + 100;

        // Step 7: 专家内部计算
        setCurrentStep('Expert Internal Computation');
        setProgress(70);
        currentY = createTitle('7. Expert Computation (Sample Token 0)', currentY);

        if (nTokens > 0) {
          const tokenIdx = 0;
          const topKIndices = getTopK(expertProbs[tokenIdx], topK);
          const tokenVector = normalizedData[tokenIdx];

          currentY = createDescription(
            `Computing expert outputs for Token 0 through selected experts: ${topKIndices.join(', ')}`,
            currentY
          );

          for (let i = 0; i < Math.min(topKIndices.length, 2); i++) {
            const expertIdx = topKIndices[i];
            currentY += 30;

            const expertComputeGroup = g.append('g')
              .attr('transform', `translate(100, ${currentY})`);

            expertComputeGroup.append('text')
              .attr('x', 0)
              .attr('y', 0)
              .attr('font-size', 13)
              .attr('font-weight', 'bold')
              .attr('fill', expertColors(expertIdx.toString()))
              .text(`Expert ${expertIdx} - Gate Weight: ${expertProbs[tokenIdx][expertIdx].toFixed(4)}`);

            // 第一层：Linear + ReLU
            const hidden = vectorMatrixMultiply(tokenVector, weights.experts[expertIdx].w1).map(relu);
            const hiddenVec = renderVector(
              hidden.slice(0, Math.min(hidden.length, 100)),
              0,
              30,
              `Hidden (after ReLU)`,
              d3.scaleSequential(d3.interpolateGreens).domain([0, Math.max(...hidden)])
            );

            await new Promise(resolve => {
              hiddenVec.cells.transition()
                .duration(600)
                .delay((d, i) => i * 2)
                .style('opacity', 1)
                .on('end', (d, i, nodes) => {
                  if (i === nodes.length - 1) resolve(true);
                });
            });

            // 第二层：Linear
            const output = vectorMatrixMultiply(hidden, weights.experts[expertIdx].w2);
            const outputVec = renderVector(
              output.slice(0, Math.min(output.length, 100)),
              0,
              60,
              `Expert Output`,
              colorScale
            );

            await new Promise(resolve => {
              outputVec.cells.transition()
                .duration(600)
                .delay((d, i) => i * 2)
                .style('opacity', 1)
                .on('end', (d, i, nodes) => {
                  if (i === nodes.length - 1) resolve(true);
                });
            });

            currentY += 100;
          }
        }

        currentY += 50;

        // Step 8: 输出加权与合并
        setCurrentStep('Weighted Output Merging');
        setProgress(85);
        currentY = createTitle('8. Output Weighting & Merging', currentY);
        currentY = createDescription(
          'Multiply each expert output by its gate weight and sum',
          currentY
        );

        // 计算所有token的最终MoE输出
        const moeOutputs: number[][] = [];
        for (let t = 0; t < nTokens; t++) {
          const topKIndices = getTopK(expertProbs[t], topK);
          const tokenVector = normalizedData[t];
          const finalOutput = Array(config.n_embd).fill(0);

          for (const expertIdx of topKIndices) {
            const hidden = vectorMatrixMultiply(tokenVector, weights.experts[expertIdx].w1).map(relu);
            const expertOut = vectorMatrixMultiply(hidden, weights.experts[expertIdx].w2);
            const gateWeight = expertProbs[t][expertIdx];

            for (let j = 0; j < finalOutput.length; j++) {
              finalOutput[j] += gateWeight * expertOut[j];
            }
          }

          moeOutputs.push(finalOutput);
        }

        const moeOutputMatrix = renderMatrix(moeOutputs, 100, currentY, 'MoE Output', colorScale);
        await new Promise(resolve => {
          moeOutputMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += moeOutputMatrix.height + matrixSpacing + 50;

        // Step 9: 残差连接
        setCurrentStep('Residual Connection');
        setProgress(92);
        currentY = createTitle('9. Residual Connection', currentY);
        currentY = createDescription(
          'Add original input to MoE output: Final = Input + MoE_Output',
          currentY
        );

        // 计算残差连接
        const finalOutput = inputData.map((row, i) =>
          row.map((val, j) => val + moeOutputs[i][j])
        );

        const finalMatrix = renderMatrix(finalOutput, 100, currentY, 'Final Output', colorScale);
        await new Promise(resolve => {
          finalMatrix.cells.transition()
            .duration(800)
            .delay((d, i) => i * 2)
            .style('opacity', 1)
            .on('end', (d, i, nodes) => {
              if (i === nodes.length - 1) resolve(true);
            });
        });

        currentY += finalMatrix.height + matrixSpacing + 50;

        // Step 10: 专家负载均衡统计
        setCurrentStep('Expert Load Balancing');
        setProgress(98);
        currentY = createTitle('10. Expert Load Balancing Statistics', currentY);

        // 统计每个专家被选中的次数
        const expertUsage = Array(nExperts).fill(0);
        for (let t = 0; t < nTokens; t++) {
          const topKIndices = getTopK(expertProbs[t], topK);
          topKIndices.forEach(idx => expertUsage[idx]++);
        }

        const usageChartWidth = Math.min(800, width - 200);
        const usageBarWidth = usageChartWidth / nExperts - 4;
        const usageChartHeight = 150;
        const usageChartX = (width - usageChartWidth) / 2;

        const usageGroup = g.append('g')
          .attr('transform', `translate(${usageChartX}, ${currentY})`);

        const maxUsage = Math.max(...expertUsage);
        const usageYScale = d3.scaleLinear()
          .domain([0, maxUsage])
          .range([0, usageChartHeight]);

        usageGroup.selectAll('.usage-bar')
          .data(expertUsage.map((count, idx) => ({ count, idx })))
          .join('rect')
          .attr('class', 'usage-bar')
          .attr('x', d => d.idx * (usageChartWidth / nExperts))
          .attr('y', usageChartHeight)
          .attr('width', usageBarWidth)
          .attr('height', 0)
          .attr('fill', d => {
            const normalized = maxUsage > 0 ? d.count / maxUsage : 0;
            return normalized < 0.3 ? '#e53e3e' : expertColors(d.idx.toString());
          })
          .attr('stroke', '#000')
          .attr('stroke-width', 1)
          .transition()
          .duration(800)
          .delay((d, i) => i * 50)
          .attr('y', d => usageChartHeight - usageYScale(d.count))
          .attr('height', d => usageYScale(d.count));

        // X轴标签
        usageGroup.selectAll('.usage-label')
          .data(expertUsage.map((_, idx) => idx))
          .join('text')
          .attr('class', 'usage-label')
          .attr('x', d => d * (usageChartWidth / nExperts) + usageBarWidth / 2)
          .attr('y', usageChartHeight + 20)
          .attr('text-anchor', 'middle')
          .attr('font-size', 11)
          .attr('fill', '#2d3748')
          .text(d => `E${d}`);

        // 使用次数标签
        usageGroup.selectAll('.usage-count')
          .data(expertUsage.map((count, idx) => ({ count, idx })))
          .join('text')
          .attr('class', 'usage-count')
          .attr('x', d => d.idx * (usageChartWidth / nExperts) + usageBarWidth / 2)
          .attr('y', d => usageChartHeight - usageYScale(d.count) - 5)
          .attr('text-anchor', 'middle')
          .attr('font-size', 10)
          .attr('font-weight', 'bold')
          .attr('fill', '#000')
          .style('opacity', 0)
          .text(d => d.count)
          .transition()
          .duration(500)
          .delay(1000)
          .style('opacity', 1);

        usageGroup.append('text')
          .attr('x', usageChartWidth / 2)
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .attr('fill', '#4a5568')
          .text('Number of times each expert was selected');

        await new Promise(resolve => setTimeout(resolve, 2000));

        // 完成
        setCurrentStep('Complete');
        setProgress(100);

        if (onComplete) {
          onComplete();
        }
      } catch (error) {
        console.error('Animation error:', error);
      }
    };

    runAnimation();

    return () => {
      const currentSvg = svgRef.current;
      if (currentSvg) {
        d3.select(currentSvg).selectAll('*').remove();
      }
    };
  }, [inputData, weights, config, tokenTexts, animationMode, onComplete]);

  return (
    <div ref={containerRef} className="w-full h-full overflow-auto bg-gray-50 rounded-lg shadow-lg p-4">
      <div className="mb-4 bg-white p-4 rounded-lg shadow">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-lg font-bold text-gray-800">MoE FFN Visualization Progress</h3>
          <span className="text-sm font-semibold text-blue-600">{progress}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
          <div
            className="bg-blue-600 h-3 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-sm text-gray-600">{currentStep}</p>
      </div>
      <svg ref={svgRef} className="w-full" />
    </div>
  );
};
