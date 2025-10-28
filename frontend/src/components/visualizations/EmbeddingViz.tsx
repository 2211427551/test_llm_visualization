'use client';

import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface EmbeddingVizProps {
  tokens: number[];
  tokenTexts: string[];
  embeddings: number[][];
  positionalEncodings: number[][];
  nEmbd: number;
  nVocab: number;
  animationMode?: 'serial' | 'parallel';
  onComplete?: () => void;
}

export const EmbeddingViz: React.FC<EmbeddingVizProps> = ({
  tokens,
  tokenTexts,
  embeddings,
  positionalEncodings,
  nEmbd,
  nVocab,
  animationMode = 'serial',
  onComplete,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [currentAnimatingToken, setCurrentAnimatingToken] = useState<number>(-1);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || embeddings.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 800;

    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .style('background', '#f8f9fa');

    const g = svg.append('g');

    const tokenY = 50;
    const matrixY = 150;
    const matrixHeight = 150;
    const embeddingVectorY = 380;
    const positionalVectorY = 480;
    const additionY = 580;
    const finalVectorY = 680;

    const blockSize = Math.min(8, Math.max(2, (width - 200) / nEmbd));
    const displayEmbd = Math.min(nEmbd, Math.floor((width - 200) / blockSize));

    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
      .domain([1, -1]);

    const tokenBlockSpacing = 10;
    const tokenBlockWidths = tokenTexts.map(token => {
      const textLength = token.length * 12;
      return Math.max(textLength + 20, 60);
    });
    const totalWidth = tokenBlockWidths.reduce((sum, w) => sum + w, 0) + 
                     (tokenBlockWidths.length - 1) * tokenBlockSpacing;
    const startX = (width - totalWidth) / 2;

    const tokenGroups = g.selectAll('.token-header')
      .data(tokenTexts.map((token, i) => ({
        token,
        id: tokens[i],
        index: i,
        width: tokenBlockWidths[i],
      })))
      .join('g')
      .attr('class', 'token-header')
      .attr('transform', (d, i) => {
        const x = startX + tokenBlockWidths.slice(0, i).reduce((sum, w) => sum + w + tokenBlockSpacing, 0);
        return `translate(${x + d.width / 2}, ${tokenY})`;
      });

    tokenGroups.each(function(d) {
      const group = d3.select(this);
      
      group.append('rect')
        .attr('x', -d.width / 2)
        .attr('y', -20)
        .attr('width', d.width)
        .attr('height', 40)
        .attr('rx', 6)
        .attr('fill', d3.interpolateBlues(0.5))
        .attr('stroke', '#2c5282')
        .attr('stroke-width', 2);

      group.append('text')
        .attr('y', 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', 14)
        .attr('font-weight', 600)
        .attr('fill', '#1a365d')
        .text(d.token);
    });

    const matrixGroup = g.append('g')
      .attr('transform', `translate(${width / 2}, ${matrixY})`);

    matrixGroup.append('rect')
      .attr('x', -120)
      .attr('y', 0)
      .attr('width', 240)
      .attr('height', matrixHeight)
      .attr('fill', '#e2e8f0')
      .attr('stroke', '#4a5568')
      .attr('stroke-width', 2)
      .attr('rx', 4);

    matrixGroup.append('text')
      .attr('y', matrixHeight / 2 - 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', 14)
      .attr('font-weight', 600)
      .attr('fill', '#2d3748')
      .text('Embedding Matrix');

    matrixGroup.append('text')
      .attr('y', matrixHeight / 2 + 10)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('fill', '#4a5568')
      .text(`${nVocab} × ${nEmbd}`);

    const animateToken = (tokenIndex: number, callback?: () => void) => {
      setCurrentAnimatingToken(tokenIndex);
      
      const tokenData = {
        token: tokenTexts[tokenIndex],
        id: tokens[tokenIndex],
        index: tokenIndex,
        width: tokenBlockWidths[tokenIndex],
      };

      const tokenX = startX + tokenBlockWidths.slice(0, tokenIndex).reduce((sum, w) => sum + w + tokenBlockSpacing, 0) + tokenData.width / 2;

      const arrow = g.append('line')
        .attr('class', 'lookup-arrow')
        .attr('x1', tokenX)
        .attr('y1', tokenY + 25)
        .attr('x2', tokenX)
        .attr('y2', tokenY + 25)
        .attr('stroke', '#3182ce')
        .attr('stroke-width', 3)
        .attr('marker-end', 'url(#arrowhead)');

      const defs = svg.append('defs');
      defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('markerWidth', 10)
        .attr('markerHeight', 10)
        .attr('refX', 5)
        .attr('refY', 3)
        .attr('orient', 'auto')
        .append('polygon')
        .attr('points', '0 0, 10 3, 0 6')
        .attr('fill', '#3182ce');

      arrow.transition()
        .duration(600)
        .attr('y2', matrixY - 5);

      setTimeout(() => {
        matrixGroup.select('rect')
          .transition()
          .duration(400)
          .attr('fill', '#fef5e7')
          .transition()
          .duration(400)
          .attr('fill', '#e2e8f0');

        const embeddingVector = embeddings[tokenIndex];
        const positionalVector = positionalEncodings[tokenIndex];

        const vectorStartX = tokenX - (displayEmbd * blockSize) / 2;

        const embeddingGroup = g.append('g')
          .attr('class', `embedding-vector-${tokenIndex}`)
          .attr('transform', `translate(${width / 2}, ${matrixY + matrixHeight + 20})`)
          .style('opacity', 0);

        embeddingGroup.append('text')
          .attr('x', 0)
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .attr('font-size', 12)
          .attr('font-weight', 600)
          .attr('fill', '#2d3748')
          .text(`Embedding "${tokenData.token}"`);

        const embeddingBlocks = embeddingGroup.selectAll('.emb-block')
          .data(embeddingVector.slice(0, displayEmbd))
          .join('g')
          .attr('class', 'emb-block')
          .attr('transform', (d, i) => `translate(${i * blockSize - (displayEmbd * blockSize) / 2}, 0)`);

        embeddingBlocks.append('rect')
          .attr('width', blockSize - 1)
          .attr('height', blockSize * 3)
          .attr('fill', d => colorScale(d))
          .attr('stroke', '#fff')
          .attr('stroke-width', 0.5);

        embeddingBlocks
          .on('mouseover', function(event, d) {
            const idx = embeddingBlocks.nodes().indexOf(this);
            d3.select(this).select('rect')
              .attr('stroke', '#000')
              .attr('stroke-width', 2);

            const tooltip = g.append('g')
              .attr('class', 'tooltip')
              .attr('transform', `translate(${event.layerX}, ${event.layerY - 30})`);

            tooltip.append('rect')
              .attr('x', -40)
              .attr('y', -20)
              .attr('width', 80)
              .attr('height', 25)
              .attr('rx', 4)
              .attr('fill', '#2d3748')
              .style('opacity', 0.95);

            tooltip.append('text')
              .attr('text-anchor', 'middle')
              .attr('y', -3)
              .attr('font-size', 11)
              .attr('fill', 'white')
              .text(`[${idx}]: ${d.toFixed(3)}`);
          })
          .on('mouseout', function() {
            d3.select(this).select('rect')
              .attr('stroke', '#fff')
              .attr('stroke-width', 0.5);
            g.selectAll('.tooltip').remove();
          });

        embeddingGroup.transition()
          .duration(600)
          .style('opacity', 1)
          .attr('transform', `translate(${vectorStartX + (displayEmbd * blockSize) / 2}, ${embeddingVectorY})`);

        setTimeout(() => {
          const positionalGroup = g.append('g')
            .attr('class', `positional-vector-${tokenIndex}`)
            .attr('transform', `translate(${vectorStartX + (displayEmbd * blockSize) / 2}, ${embeddingVectorY})`)
            .style('opacity', 0);

          positionalGroup.append('text')
            .attr('x', 0)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('font-size', 12)
            .attr('font-weight', 600)
            .attr('fill', '#2d3748')
            .text(`Position ${tokenIndex}`);

          const positionalBlocks = positionalGroup.selectAll('.pos-block')
            .data(positionalVector.slice(0, displayEmbd))
            .join('g')
            .attr('class', 'pos-block')
            .attr('transform', (d, i) => `translate(${i * blockSize - (displayEmbd * blockSize) / 2}, 0)`);

          positionalBlocks.append('rect')
            .attr('width', blockSize - 1)
            .attr('height', blockSize * 3)
            .attr('fill', d => d3.interpolateGreens(Math.abs(d)))
            .attr('stroke', '#fff')
            .attr('stroke-width', 0.5);

          positionalBlocks
            .on('mouseover', function(event, d) {
              const idx = positionalBlocks.nodes().indexOf(this);
              d3.select(this).select('rect')
                .attr('stroke', '#000')
                .attr('stroke-width', 2);

              const tooltip = g.append('g')
                .attr('class', 'tooltip')
                .attr('transform', `translate(${event.layerX}, ${event.layerY - 30})`);

              tooltip.append('rect')
                .attr('x', -40)
                .attr('y', -20)
                .attr('width', 80)
                .attr('height', 25)
                .attr('rx', 4)
                .attr('fill', '#2d3748')
                .style('opacity', 0.95);

              tooltip.append('text')
                .attr('text-anchor', 'middle')
                .attr('y', -3)
                .attr('font-size', 11)
                .attr('fill', 'white')
                .text(`[${idx}]: ${d.toFixed(3)}`);
            })
            .on('mouseout', function() {
              d3.select(this).select('rect')
                .attr('stroke', '#fff')
                .attr('stroke-width', 0.5);
              g.selectAll('.tooltip').remove();
            });

          positionalGroup.transition()
            .duration(600)
            .style('opacity', 1)
            .attr('transform', `translate(${vectorStartX + (displayEmbd * blockSize) / 2}, ${positionalVectorY})`);

          setTimeout(() => {
            const plusSign = g.append('text')
              .attr('x', vectorStartX + (displayEmbd * blockSize) / 2)
              .attr('y', additionY)
              .attr('text-anchor', 'middle')
              .attr('font-size', 36)
              .attr('font-weight', 'bold')
              .attr('fill', '#48bb78')
              .style('opacity', 0)
              .text('+');

            plusSign.transition()
              .duration(400)
              .style('opacity', 1);

            setTimeout(() => {
              embeddingGroup.transition()
                .duration(800)
                .attr('transform', `translate(${vectorStartX + (displayEmbd * blockSize) / 2}, ${finalVectorY - 30})`);

              positionalGroup.transition()
                .duration(800)
                .attr('transform', `translate(${vectorStartX + (displayEmbd * blockSize) / 2}, ${finalVectorY - 30})`);

              plusSign.transition()
                .duration(800)
                .style('opacity', 0);

              setTimeout(() => {
                embeddingGroup.remove();
                positionalGroup.remove();
                plusSign.remove();

                const finalVector = embeddingVector.map((e, i) => e + positionalVector[i]);

                const finalGroup = g.append('g')
                  .attr('class', `final-vector-${tokenIndex}`)
                  .attr('transform', `translate(${vectorStartX + (displayEmbd * blockSize) / 2}, ${finalVectorY})`)
                  .style('opacity', 0);

                finalGroup.append('text')
                  .attr('x', 0)
                  .attr('y', -10)
                  .attr('text-anchor', 'middle')
                  .attr('font-size', 12)
                  .attr('font-weight', 600)
                  .attr('fill', '#2d3748')
                  .text(`Input Vector "${tokenData.token}"`);

                const finalBlocks = finalGroup.selectAll('.final-block')
                  .data(finalVector.slice(0, displayEmbd))
                  .join('g')
                  .attr('class', 'final-block')
                  .attr('transform', (d, i) => `translate(${i * blockSize - (displayEmbd * blockSize) / 2}, 0)`);

                finalBlocks.append('rect')
                  .attr('width', blockSize - 1)
                  .attr('height', blockSize * 3)
                  .attr('fill', d => colorScale(d))
                  .attr('stroke', '#fff')
                  .attr('stroke-width', 0.5);

                finalBlocks
                  .on('mouseover', function(event, d) {
                    const idx = finalBlocks.nodes().indexOf(this);
                    d3.select(this).select('rect')
                      .attr('stroke', '#000')
                      .attr('stroke-width', 2);

                    const tooltip = g.append('g')
                      .attr('class', 'tooltip')
                      .attr('transform', `translate(${event.layerX}, ${event.layerY - 40})`);

                    tooltip.append('rect')
                      .attr('x', -70)
                      .attr('y', -25)
                      .attr('width', 140)
                      .attr('height', 30)
                      .attr('rx', 4)
                      .attr('fill', '#2d3748')
                      .style('opacity', 0.95);

                    tooltip.append('text')
                      .attr('text-anchor', 'middle')
                      .attr('y', -6)
                      .attr('font-size', 11)
                      .attr('fill', 'white')
                      .text(`Emb + Pos`);

                    tooltip.append('text')
                      .attr('text-anchor', 'middle')
                      .attr('y', 8)
                      .attr('font-size', 10)
                      .attr('fill', 'white')
                      .text(`[${idx}]: ${d.toFixed(3)}`);
                  })
                  .on('mouseout', function() {
                    d3.select(this).select('rect')
                      .attr('stroke', '#fff')
                      .attr('stroke-width', 0.5);
                    g.selectAll('.tooltip').remove();
                  });

                finalGroup.transition()
                  .duration(600)
                  .style('opacity', 1)
                  .on('end', () => {
                    arrow.remove();
                    if (callback) callback();
                  });
              }, 800);
            }, 500);
          }, 700);
        }, 700);
      }, 600);
    };

    if (animationMode === 'serial') {
      const animateNext = (index: number) => {
        if (index < tokens.length) {
          animateToken(index, () => {
            setTimeout(() => animateNext(index + 1), 500);
          });
        } else {
          setCurrentAnimatingToken(-1);
          if (onComplete) {
            setTimeout(() => onComplete(), 1000);
          }
        }
      };
      animateNext(0);
    } else {
      tokens.forEach((token, index) => {
        setTimeout(() => {
          animateToken(index, () => {
            if (index === tokens.length - 1 && onComplete) {
              setCurrentAnimatingToken(-1);
              setTimeout(() => onComplete(), 1000);
            }
          });
        }, index * 300);
      });
    }

    return () => {
      const svg = svgRef.current;
      if (svg) {
        d3.select(svg).selectAll('*').remove();
      }
    };
  }, [tokens, tokenTexts, embeddings, positionalEncodings, nEmbd, nVocab, animationMode, onComplete]);

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef}></svg>
      {currentAnimatingToken >= 0 && (
        <div className="text-center text-sm text-gray-600 mt-2">
          正在处理词元: {tokenTexts[currentAnimatingToken]} (位置 {currentAnimatingToken})
        </div>
      )}
    </div>
  );
};
