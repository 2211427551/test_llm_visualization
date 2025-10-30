'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { drawMatrix, addLabels } from '@/lib/visualization/d3-helpers';

interface AttentionMatrixProps {
  attentionWeights: number[][];
  tokens: string[];
  title?: string;
  onHover?: (source: number, target: number, value: number) => void;
  showValues?: boolean;
  animated?: boolean;
}

export function AttentionMatrix({ 
  attentionWeights, 
  tokens, 
  title,
  onHover,
  showValues = false,
  animated = true,
}: AttentionMatrixProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number; value: number } | null>(null);
  
  useEffect(() => {
    if (!svgRef.current || !attentionWeights.length) return;
    
    const svg = d3.select(svgRef.current);
    const cellSize = 40;
    const margin = { top: 40, right: 20, bottom: 40, left: 100 };
    
    // Clear previous content
    svg.selectAll('*').remove();
    
    // Create main group with margins
    const g = svg.append('g')
      .attr('transform', `translate(${margin.left}, ${margin.top})`);
    
    // Draw matrix
    const gSelection = g.node() ? d3.select(g.node()!) : svg;
    drawMatrix(gSelection as any, attentionWeights, {
      cellSize,
      showValues,
      animationDuration: animated ? 500 : 0,
      onCellHover: (i: number, j: number, value: number) => {
        setHoveredCell({ i, j, value });
        onHover?.(i, j, value);
      },
    });
    
    // Add labels
    addLabels(gSelection as any, {
      x: tokens,
      y: tokens,
    }, {
      cellSize,
      fontSize: 12,
      offset: 15,
    });
    
    // Add title
    if (title) {
      svg.append('text')
        .attr('x', margin.left + (tokens.length * cellSize) / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('font-weight', '600')
        .attr('fill', '#ffffff')
        .text(title);
    }
    
  }, [attentionWeights, tokens, title, onHover, showValues, animated]);
  
  const width = tokens.length * 40 + 120;
  const height = tokens.length * 40 + 80;
  
  return (
    <div className="relative bg-slate-900 rounded-xl p-6">
      <svg 
        ref={svgRef} 
        width={width} 
        height={height}
        className="mx-auto"
      />
      
      {hoveredCell && (
        <div className="absolute top-4 right-4 bg-black/80 text-white px-4 py-2 rounded-lg text-sm">
          <div className="font-mono">
            <div>From: <span className="text-cyan-400">{tokens[hoveredCell.i]}</span></div>
            <div>To: <span className="text-pink-400">{tokens[hoveredCell.j]}</span></div>
            <div>Weight: <span className="text-purple-400">{hoveredCell.value.toFixed(4)}</span></div>
          </div>
        </div>
      )}
      
      <div className="mt-4 flex items-center justify-center gap-4 text-sm text-slate-400">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-gradient-to-r from-blue-500 to-purple-500"></div>
          <span>Low attention</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded bg-gradient-to-r from-purple-500 to-pink-500"></div>
          <span>High attention</span>
        </div>
      </div>
    </div>
  );
}
