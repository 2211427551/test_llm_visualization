'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface CellData {
  value: number;
  i: number;
  j: number;
}

interface AttentionMatrixProps {
  attentionWeights: number[][];
  tokens: string[];
  title?: string;
  onHover?: (i: number, j: number, value: number) => void;
  onCellClick?: (i: number, j: number, value: number) => void;
  showValues?: boolean;
  animated?: boolean;
}

export const AttentionMatrix: React.FC<AttentionMatrixProps> = ({
  attentionWeights,
  tokens,
  title,
  onHover,
  onCellClick,
  showValues = false,
  animated = true,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredCell, setHoveredCell] = useState<CellData | null>(null);

  useEffect(() => {
    if (!svgRef.current || !attentionWeights.length) return;

    const svg = d3.select(svgRef.current);
    const width = 600;
    const height = 600;
    const padding = 50;

    // 清空之前的内容
    svg.selectAll('*').remove();

    // 计算单元格大小
    const cellSize = Math.min(
      (width - padding * 2) / tokens.length,
      (height - padding * 2) / tokens.length
    );

    // 创建颜色比例尺
    const maxValue = d3.max(attentionWeights.flat()) || 1;
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, maxValue]);

    // 创建主组
    const g = svg.append('g')
      .attr('transform', `translate(${padding}, ${padding})`);

    // 扁平化数据
    const cellData: CellData[] = attentionWeights.flatMap((row, i) =>
      row.map((value, j) => ({ value, i, j }))
    );

    // 创建单元格
    const cells = g.selectAll<SVGGElement, CellData>('.matrix-cell')
      .data(cellData)
      .join('g')
      .attr('class', 'matrix-cell')
      .attr('transform', (d: CellData) => `translate(${d.j * cellSize}, ${d.i * cellSize})`);

    // 绘制矩形
    cells.append('rect')
      .attr('width', cellSize - 1)
      .attr('height', cellSize - 1)
      .attr('fill', (d: CellData) => colorScale(d.value))
      .attr('stroke', 'none')
      .attr('opacity', animated ? 0 : 1)
      .style('cursor', onCellClick ? 'pointer' : 'default')
      .on('click', function(event: MouseEvent, d: CellData) {
        if (onCellClick) onCellClick(d.i, d.j, d.value);
      })
      .on('mouseenter', function(event: MouseEvent, d: CellData) {
        setHoveredCell(d);
        if (onHover) onHover(d.i, d.j, d.value);
        
        // 高亮效果
        d3.select(this)
          .attr('stroke', 'white')
          .attr('stroke-width', 2);
        
        // 高亮行列
        cells.selectAll('rect')
          .attr('opacity', function(cell: any) {
            return cell.i === d.i || cell.j === d.j ? 1 : 0.3;
          });
      })
      .on('mouseleave', function(event: MouseEvent, d: CellData) {
        d3.select(this)
          .attr('stroke', 'none');
        
        // 恢复透明度
        cells.selectAll('rect').attr('opacity', 1);
      });

    // 动画效果
    if (animated) {
      cells.selectAll('rect')
        .transition()
        .duration(500)
        .delay((_d: any, i: number) => i * 5)
        .attr('opacity', 1);
    }

    // 添加数值文本（如果需要）
    if (showValues) {
      cells.append('text')
        .attr('x', cellSize / 2)
        .attr('y', cellSize / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', Math.min(cellSize / 3, 10))
        .attr('fill', 'white')
        .attr('pointer-events', 'none')
        .text((d: CellData) => d.value.toFixed(2));
    }

    // 添加X轴标签
    g.selectAll('.label-x')
      .data(tokens)
      .join('text')
      .attr('class', 'label-x')
      .attr('x', (d: string, i: number) => i * cellSize + cellSize / 2)
      .attr('y', tokens.length * cellSize + 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', Math.min(cellSize / 2, 12))
      .attr('fill', '#cbd5e1')
      .text((d: string) => d);

    // 添加Y轴标签
    g.selectAll('.label-y')
      .data(tokens)
      .join('text')
      .attr('class', 'label-y')
      .attr('x', -10)
      .attr('y', (d: string, i: number) => i * cellSize + cellSize / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', Math.min(cellSize / 2, 12))
      .attr('fill', '#cbd5e1')
      .text((d: string) => d);

  }, [attentionWeights, tokens, onHover, onCellClick, showValues, animated]);

  return (
    <div ref={containerRef} className="relative">
      {title && (
        <h3 className="text-white text-lg font-semibold mb-4">{title}</h3>
      )}
      <svg
        ref={svgRef}
        width={700}
        height={700}
        className="bg-slate-900 rounded-lg"
      />
      
      {/* 悬停信息 */}
      {hoveredCell && (
        <div className="absolute top-4 right-4 bg-black/80 text-white px-4 py-2 rounded text-sm z-10">
          <div className="font-semibold mb-1">Attention Weight</div>
          <div>From: {tokens[hoveredCell.i]}</div>
          <div>To: {tokens[hoveredCell.j]}</div>
          <div>Value: {hoveredCell.value.toFixed(4)}</div>
        </div>
      )}
      
      {/* 颜色图例 */}
      <div className="mt-4 flex items-center justify-center gap-2 text-sm text-slate-400">
        <span>Low</span>
        <div className="w-32 h-4 rounded" 
             style={{
               background: 'linear-gradient(to right, #440154, #31688e, #35b779, #fde724)'
             }}
        />
        <span>High</span>
      </div>
    </div>
  );
};
