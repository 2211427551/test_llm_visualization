'use client';

import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface TokenizationVizProps {
  text: string;
  tokens: number[];
  tokenTexts: string[];
  onComplete?: () => void;
}

export const TokenizationViz: React.FC<TokenizationVizProps> = ({
  text,
  tokens,
  tokenTexts,
  onComplete,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || tokenTexts.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 400;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', '#f8f9fa');

    const g = svg.append('g');

    const originalTextY = 80;
    const tokenBlockY = 200;
    const tokenBlockHeight = 50;
    const tokenBlockSpacing = 10;
    const tokenIdY = tokenBlockY + tokenBlockHeight + 25;

    const originalText = g.append('text')
      .attr('x', width / 2)
      .attr('y', originalTextY)
      .attr('text-anchor', 'middle')
      .attr('font-size', 28)
      .attr('font-weight', 'bold')
      .attr('fill', '#2c3e50')
      .text(text)
      .style('opacity', 1);

    setTimeout(() => {
      originalText
        .transition()
        .duration(800)
        .ease(d3.easeCubicInOut)
        .style('opacity', 0)
        .on('end', () => {
          const tokenBlockWidths = tokenTexts.map(token => {
            const textLength = token.length * 12;
            return Math.max(textLength + 20, 60);
          });

          const totalWidth = tokenBlockWidths.reduce((sum, w) => sum + w, 0) + 
                           (tokenBlockWidths.length - 1) * tokenBlockSpacing;
          const startX = (width - totalWidth) / 2;

          const tokenGroups = g.selectAll('.token-group')
            .data(tokenTexts.map((token, i) => ({
              token,
              id: tokens[i],
              index: i,
              width: tokenBlockWidths[i],
            })))
            .join('g')
            .attr('class', 'token-group')
            .attr('transform', (d, i) => {
              const x = startX + tokenBlockWidths.slice(0, i).reduce((sum, w) => sum + w + tokenBlockSpacing, 0);
              return `translate(${x + d.width / 2}, ${originalTextY})`;
            })
            .style('opacity', 0);

          tokenGroups.each(function(d) {
            const group = d3.select(this);
            
            group.append('rect')
              .attr('class', 'token-rect')
              .attr('x', -d.width / 2)
              .attr('y', -tokenBlockHeight / 2)
              .attr('width', d.width)
              .attr('height', tokenBlockHeight)
              .attr('rx', 8)
              .attr('fill', d3.interpolateBlues(0.5 + d.index * 0.1 / tokenTexts.length))
              .attr('stroke', '#2c5282')
              .attr('stroke-width', 2)
              .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))');

            group.append('text')
              .attr('class', 'token-text')
              .attr('y', 5)
              .attr('text-anchor', 'middle')
              .attr('font-size', 16)
              .attr('font-weight', 600)
              .attr('fill', '#1a365d')
              .text(d.token);
          });

          tokenGroups
            .transition()
            .duration(600)
            .delay((d, i) => i * 100)
            .ease(d3.easeCubicOut)
            .style('opacity', 1)
            .attr('transform', (d, i) => {
              const x = startX + tokenBlockWidths.slice(0, i).reduce((sum, w) => sum + w + tokenBlockSpacing, 0);
              return `translate(${x + d.width / 2}, ${tokenBlockY})`;
            })
            .on('end', (d, i) => {
              if (i === tokenTexts.length - 1) {
                const tokenIdGroups = g.selectAll('.token-id')
                  .data(tokenTexts.map((token, i) => ({
                    token,
                    id: tokens[i],
                    index: i,
                    width: tokenBlockWidths[i],
                  })))
                  .join('g')
                  .attr('class', 'token-id')
                  .attr('transform', (d, i) => {
                    const x = startX + tokenBlockWidths.slice(0, i).reduce((sum, w) => sum + w + tokenBlockSpacing, 0);
                    return `translate(${x + d.width / 2}, ${tokenIdY})`;
                  })
                  .style('opacity', 0);

                tokenIdGroups.append('text')
                  .attr('text-anchor', 'middle')
                  .attr('font-size', 12)
                  .attr('fill', '#718096')
                  .text(d => `ID: ${d.id}`);

                tokenIdGroups
                  .transition()
                  .duration(400)
                  .delay((d, i) => i * 50)
                  .style('opacity', 1)
                  .on('end', (d, i) => {
                    if (i === tokenTexts.length - 1 && onComplete) {
                      setTimeout(() => onComplete(), 500);
                    }
                  });
              }
            });

          tokenGroups
            .on('mouseover', function(event, d) {
              d3.select(this).select('.token-rect')
                .transition()
                .duration(200)
                .attr('fill', d3.interpolateBlues(0.7))
                .attr('stroke-width', 3);

              const tooltip = g.append('g')
                .attr('class', 'tooltip')
                .attr('transform', `translate(${event.layerX}, ${event.layerY - 40})`);

              tooltip.append('rect')
                .attr('x', -60)
                .attr('y', -25)
                .attr('width', 120)
                .attr('height', 30)
                .attr('rx', 5)
                .attr('fill', '#2d3748')
                .style('opacity', 0.9);

              tooltip.append('text')
                .attr('text-anchor', 'middle')
                .attr('y', -5)
                .attr('font-size', 12)
                .attr('fill', 'white')
                .text(`"${d.token}" [${d.id}] pos:${d.index}`);
            })
            .on('mouseout', function(event, d) {
              d3.select(this).select('.token-rect')
                .transition()
                .duration(200)
                .attr('fill', d3.interpolateBlues(0.5 + d.index * 0.1 / tokenTexts.length))
                .attr('stroke-width', 2);

              g.selectAll('.tooltip').remove();
            });
        });
    }, 500);

    return () => {
      const svg = svgRef.current;
      if (svg) {
        d3.select(svg).selectAll('*').remove();
      }
    };
  }, [text, tokens, tokenTexts, onComplete]);

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef}></svg>
    </div>
  );
};
