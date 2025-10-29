'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface OutputLayerVizProps {
  finalHiddenState: number[][];
  vocabulary: string[];
  onComplete?: () => void;
}

interface LogitData {
  tokenId: number;
  token: string;
  logit: number;
  probability: number;
}

export const OutputLayerViz: React.FC<OutputLayerVizProps> = ({
  finalHiddenState,
  vocabulary,
  onComplete,
}) => {
  const [currentStage, setCurrentStage] = useState<
    'layernorm' | 'selection' | 'logits' | 'softmax' | 'prediction'
  >('layernorm');
  const [topKLogits, setTopKLogits] = useState<LogitData[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    const stages: Array<'layernorm' | 'selection' | 'logits' | 'softmax' | 'prediction'> = [
      'layernorm',
      'selection',
      'logits',
      'softmax',
      'prediction',
    ];
    let currentIndex = 0;

    const interval = setInterval(() => {
      currentIndex++;
      if (currentIndex < stages.length) {
        setCurrentStage(stages[currentIndex]);
      } else {
        clearInterval(interval);
        onComplete?.();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [onComplete]);

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-4">输出层可视化</h3>

        {currentStage === 'layernorm' && (
          <FinalLayerNormViz hiddenState={finalHiddenState} />
        )}

        {currentStage === 'selection' && (
          <TokenSelectionViz hiddenState={finalHiddenState} />
        )}

        {currentStage === 'logits' && (
          <LogitsHeadViz
            selectedVector={finalHiddenState[finalHiddenState.length - 1]}
            vocabulary={vocabulary}
            onLogitsComputed={setTopKLogits}
          />
        )}

        {currentStage === 'softmax' && (
          <SoftmaxViz logits={topKLogits} />
        )}

        {currentStage === 'prediction' && (
          <PredictionDisplay predictions={topKLogits} searchTerm={searchTerm} />
        )}
      </div>

      {(currentStage === 'logits' || currentStage === 'softmax' || currentStage === 'prediction') && (
        <div className="bg-white rounded-lg shadow-md p-4">
          <input
            type="text"
            placeholder="搜索token..."
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      )}

      {currentStage === 'prediction' && <TopKPredictions predictions={topKLogits} />}
    </div>
  );
};

const FinalLayerNormViz: React.FC<{ hiddenState: number[][] }> = ({ hiddenState }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !hiddenState.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 300;
    const margin = { top: 40, right: 40, bottom: 40, left: 60 };

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('text')
      .attr('x', (width - margin.left - margin.right) / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-lg font-bold fill-blue-600')
      .text('Final Layer Normalization');

    const embDim = hiddenState[0].length;
    const cellSize = Math.min(20, (width - margin.left - margin.right) / embDim);

    hiddenState.forEach((vector, i) => {
      const y = i * 30;
      vector.slice(0, 20).forEach((val, j) => {
        const normalizedVal = Math.tanh(val);
        g.append('rect')
          .attr('x', j * cellSize)
          .attr('y', y)
          .attr('width', cellSize - 2)
          .attr('height', 25)
          .attr('fill', d3.interpolateRdBu(0.5 - normalizedVal * 0.5))
          .attr('opacity', 0)
          .transition()
          .duration(500)
          .attr('opacity', 0.8);
      });

      g.append('text')
        .attr('x', -10)
        .attr('y', y + 17)
        .attr('text-anchor', 'end')
        .attr('class', 'text-xs fill-gray-600')
        .text(`T${i}`);
    });
  }, [hiddenState]);

  return (
    <div>
      <p className="text-gray-700 mb-4">对最后一个Transformer Block的输出进行Layer Normalization</p>
      <svg ref={svgRef}></svg>
    </div>
  );
};

const TokenSelectionViz: React.FC<{ hiddenState: number[][] }> = ({ hiddenState }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !hiddenState.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 250;
    const margin = { top: 40, right: 40, bottom: 40, left: 60 };

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('text')
      .attr('x', (width - margin.left - margin.right) / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-lg font-bold fill-green-600')
      .text('Token Selection (Next Token Prediction)');

    const seqLen = hiddenState.length;
    const boxWidth = 80;
    const spacing = 20;
    const totalWidth = seqLen * (boxWidth + spacing);
    const startX = ((width - margin.left - margin.right) - totalWidth) / 2;

    hiddenState.forEach((vector, i) => {
      const x = startX + i * (boxWidth + spacing);
      const isLast = i === seqLen - 1;

      const box = g.append('rect')
        .attr('x', x)
        .attr('y', 50)
        .attr('width', boxWidth)
        .attr('height', 60)
        .attr('fill', isLast ? '#10b981' : '#d1d5db')
        .attr('stroke', isLast ? '#059669' : '#9ca3af')
        .attr('stroke-width', isLast ? 3 : 1)
        .attr('rx', 5)
        .attr('opacity', 0.3);

      if (isLast) {
        box.transition()
          .duration(800)
          .attr('opacity', 1)
          .attr('filter', 'drop-shadow(0 0 10px #10b981)');
      }

      g.append('text')
        .attr('x', x + boxWidth / 2)
        .attr('y', 80)
        .attr('text-anchor', 'middle')
        .attr('class', `text-sm font-semibold ${isLast ? 'fill-white' : 'fill-gray-600'}`)
        .text(`Token ${i}`);

      if (isLast) {
        g.append('text')
          .attr('x', x + boxWidth / 2)
          .attr('y', 140)
          .attr('text-anchor', 'middle')
          .attr('class', 'text-xs font-bold fill-green-700')
          .text('← 用于预测');
      }
    });
  }, [hiddenState]);

  return (
    <div>
      <p className="text-gray-700 mb-4">
        对于Next Token Prediction任务，我们使用最后一个token的输出向量进行预测
      </p>
      <svg ref={svgRef}></svg>
    </div>
  );
};

const LogitsHeadViz: React.FC<{
  selectedVector: number[];
  vocabulary: string[];
  onLogitsComputed: (logits: LogitData[]) => void;
}> = ({ selectedVector, vocabulary, onLogitsComputed }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !selectedVector.length) return;

    const nVocab = vocabulary.length;
    const logits: number[] = [];

    for (let i = 0; i < nVocab; i++) {
      let logit = 0;
      for (let j = 0; j < Math.min(selectedVector.length, 50); j++) {
        logit += selectedVector[j] * (Math.random() - 0.5) * 2;
      }
      logits.push(logit);
    }

    const logitDataArray: LogitData[] = logits.map((logit, i) => ({
      tokenId: i,
      token: vocabulary[i] || `token_${i}`,
      logit,
      probability: 0,
    }));

    logitDataArray.sort((a, b) => b.logit - a.logit);
    const topK = logitDataArray.slice(0, 20);

    const expLogits = topK.map((d) => Math.exp(d.logit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    topK.forEach((d, i) => {
      d.probability = expLogits[i] / sumExp;
    });

    onLogitsComputed(topK);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 900;
    const height = 400;
    const margin = { top: 40, right: 40, bottom: 80, left: 60 };

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('text')
      .attr('x', (width - margin.left - margin.right) / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-lg font-bold fill-orange-600')
      .text(`Logits Head: 投影到词汇表空间 (Top-20)`);

    const xScale = d3
      .scaleBand()
      .domain(topK.map((d) => d.token))
      .range([0, width - margin.left - margin.right])
      .padding(0.2);

    const yScale = d3
      .scaleLinear()
      .domain([Math.min(0, d3.min(topK, (d) => d.logit) || 0), d3.max(topK, (d) => d.logit) || 1])
      .range([height - margin.top - margin.bottom, 0]);

    const colorScale = (val: number) => (val > 0 ? '#f97316' : '#3b82f6');

    const bars = g
      .selectAll('.logit-bar')
      .data(topK)
      .join('rect')
      .attr('class', 'logit-bar')
      .attr('x', (d) => xScale(d.token) || 0)
      .attr('y', yScale(0))
      .attr('width', xScale.bandwidth())
      .attr('height', 0)
      .attr('fill', (d) => colorScale(d.logit))
      .attr('opacity', 0.8);

    bars
      .transition()
      .duration(1000)
      .attr('y', (d) => Math.min(yScale(d.logit), yScale(0)))
      .attr('height', (d) => Math.abs(yScale(d.logit) - yScale(0)));

    bars
      .on('mouseover', function (event, d) {
        d3.select(this).attr('opacity', 1).attr('stroke', '#000').attr('stroke-width', 2);

        const tooltip = g
          .append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${xScale(d.token)! + xScale.bandwidth() / 2}, ${yScale(d.logit) - 10})`);

        tooltip
          .append('rect')
          .attr('x', -60)
          .attr('y', -40)
          .attr('width', 120)
          .attr('height', 35)
          .attr('fill', '#1f2937')
          .attr('rx', 5)
          .attr('opacity', 0.9);

        tooltip
          .append('text')
          .attr('y', -25)
          .attr('text-anchor', 'middle')
          .attr('class', 'text-xs fill-white font-semibold')
          .text(d.token);

        tooltip
          .append('text')
          .attr('y', -10)
          .attr('text-anchor', 'middle')
          .attr('class', 'text-xs fill-yellow-300')
          .text(`Logit: ${d.logit.toFixed(2)}`);
      })
      .on('mouseout', function () {
        d3.select(this).attr('opacity', 0.8).attr('stroke', 'none');
        g.selectAll('.tooltip').remove();
      });

    g.selectAll('.token-label')
      .data(topK)
      .join('text')
      .attr('class', 'token-label')
      .attr('x', (d) => (xScale(d.token) || 0) + xScale.bandwidth() / 2)
      .attr('y', height - margin.top - margin.bottom + 20)
      .attr('text-anchor', 'end')
      .attr('transform', (d) => {
        const x = (xScale(d.token) || 0) + xScale.bandwidth() / 2;
        const y = height - margin.top - margin.bottom + 20;
        return `rotate(-45, ${x}, ${y})`;
      })
      .attr('class', 'text-xs fill-gray-700')
      .text((d) => d.token);

    g.append('g')
      .attr('transform', `translate(0,${yScale(0)})`)
      .call(d3.axisBottom(xScale).tickSize(0).tickFormat(() => ''))
      .select('.domain')
      .attr('stroke', '#9ca3af');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('class', 'fill-gray-600');

    g.append('text')
      .attr('x', -margin.top)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, -${margin.top}, -40)`)
      .attr('class', 'text-sm fill-gray-700 font-semibold')
      .text('Logit值');
  }, [selectedVector, vocabulary, onLogitsComputed]);

  return (
    <div>
      <p className="text-gray-700 mb-4">
        通过线性层将向量投影到词汇表空间，生成每个token的未归一化分数（logits）
      </p>
      <svg ref={svgRef}></svg>
    </div>
  );
};

const SoftmaxViz: React.FC<{ logits: LogitData[] }> = ({ logits }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !logits.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 900;
    const height = 400;
    const margin = { top: 40, right: 40, bottom: 80, left: 60 };

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    g.append('text')
      .attr('x', (width - margin.left - margin.right) / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('class', 'text-lg font-bold fill-purple-600')
      .text('Softmax归一化: Logits → Probabilities');

    const xScale = d3
      .scaleBand()
      .domain(logits.map((d) => d.token))
      .range([0, width - margin.left - margin.right])
      .padding(0.2);

    const yScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([height - margin.top - margin.bottom, 0]);

    const colorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([0, d3.max(logits, (d) => d.probability) || 1]);

    const bars = g
      .selectAll('.prob-bar')
      .data(logits)
      .join('rect')
      .attr('class', 'prob-bar')
      .attr('x', (d) => xScale(d.token) || 0)
      .attr('y', yScale(0))
      .attr('width', xScale.bandwidth())
      .attr('height', 0)
      .attr('fill', (d) => colorScale(d.probability))
      .attr('opacity', 0.8);

    bars
      .transition()
      .duration(1500)
      .attr('y', (d) => yScale(d.probability))
      .attr('height', (d) => yScale(0) - yScale(d.probability));

    bars
      .on('mouseover', function (event, d) {
        d3.select(this).attr('opacity', 1).attr('stroke', '#000').attr('stroke-width', 2);

        const tooltip = g
          .append('g')
          .attr('class', 'tooltip')
          .attr('transform', `translate(${xScale(d.token)! + xScale.bandwidth() / 2}, ${yScale(d.probability) - 10})`);

        tooltip
          .append('rect')
          .attr('x', -70)
          .attr('y', -50)
          .attr('width', 140)
          .attr('height', 45)
          .attr('fill', '#1f2937')
          .attr('rx', 5)
          .attr('opacity', 0.9);

        tooltip
          .append('text')
          .attr('y', -30)
          .attr('text-anchor', 'middle')
          .attr('class', 'text-xs fill-white font-semibold')
          .text(d.token);

        tooltip
          .append('text')
          .attr('y', -15)
          .attr('text-anchor', 'middle')
          .attr('class', 'text-xs fill-green-300')
          .text(`概率: ${(d.probability * 100).toFixed(1)}%`);

        tooltip
          .append('text')
          .attr('y', -0)
          .attr('text-anchor', 'middle')
          .attr('class', 'text-xs fill-yellow-300')
          .text(`Logit: ${d.logit.toFixed(2)}`);
      })
      .on('mouseout', function () {
        d3.select(this).attr('opacity', 0.8).attr('stroke', 'none');
        g.selectAll('.tooltip').remove();
      });

    g.selectAll('.token-label')
      .data(logits)
      .join('text')
      .attr('class', 'token-label')
      .attr('x', (d) => (xScale(d.token) || 0) + xScale.bandwidth() / 2)
      .attr('y', height - margin.top - margin.bottom + 20)
      .attr('text-anchor', 'end')
      .attr('transform', (d) => {
        const x = (xScale(d.token) || 0) + xScale.bandwidth() / 2;
        const y = height - margin.top - margin.bottom + 20;
        return `rotate(-45, ${x}, ${y})`;
      })
      .attr('class', 'text-xs fill-gray-700')
      .text((d) => d.token);

    g.append('g')
      .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
      .call(d3.axisBottom(xScale).tickSize(0).tickFormat(() => ''))
      .select('.domain')
      .attr('stroke', '#9ca3af');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => `${(Number(d) * 100).toFixed(0)}%`))
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('class', 'fill-gray-600');

    g.append('text')
      .attr('x', -margin.top)
      .attr('y', -40)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, -${margin.top}, -40)`)
      .attr('class', 'text-sm fill-gray-700 font-semibold')
      .text('概率');
  }, [logits]);

  return (
    <div>
      <p className="text-gray-700 mb-4">
        应用Softmax函数将logits转换为概率分布，所有概率和为1
      </p>
      <svg ref={svgRef}></svg>
    </div>
  );
};

const PredictionDisplay: React.FC<{ predictions: LogitData[]; searchTerm: string }> = ({
  predictions,
  searchTerm,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);

  const filteredPredictions = searchTerm
    ? predictions.filter((p) => p.token.toLowerCase().includes(searchTerm.toLowerCase()))
    : predictions;

  const topPrediction = filteredPredictions.length > 0 ? filteredPredictions[0] : null;

  useEffect(() => {
    if (!svgRef.current || !filteredPredictions.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 900;
    const height = 400;
    const margin = { top: 60, right: 40, bottom: 80, left: 60 };

    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3
      .scaleBand()
      .domain(filteredPredictions.map((d) => d.token))
      .range([0, width - margin.left - margin.right])
      .padding(0.2);

    const yScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([height - margin.top - margin.bottom, 0]);

    const colorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([0, d3.max(filteredPredictions, (d) => d.probability) || 1]);

    const bars = g
      .selectAll('.pred-bar')
      .data(filteredPredictions)
      .join('rect')
      .attr('class', 'pred-bar')
      .attr('x', (d) => xScale(d.token) || 0)
      .attr('y', (d) => yScale(d.probability))
      .attr('width', xScale.bandwidth())
      .attr('height', (d) => yScale(0) - yScale(d.probability))
      .attr('fill', (d, i) => (i === 0 ? '#fbbf24' : colorScale(d.probability)))
      .attr('opacity', 0.8);

    bars.filter((d, i) => i === 0)
      .transition()
      .duration(800)
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 3)
      .attr('opacity', 1)
      .style('filter', 'drop-shadow(0 0 15px #fbbf24)');

    bars
      .on('mouseover', function () {
        d3.select(this).attr('opacity', 1).attr('stroke', '#000').attr('stroke-width', 2);
      })
      .on('mouseout', function (_event, d) {
        const idx = filteredPredictions.indexOf(d);
        d3.select(this)
          .attr('opacity', 0.8)
          .attr('stroke', idx === 0 ? '#f59e0b' : 'none')
          .attr('stroke-width', idx === 0 ? 3 : 0);
      });

    g.selectAll('.prob-label')
      .data(filteredPredictions)
      .join('text')
      .attr('class', 'prob-label')
      .attr('x', (d) => (xScale(d.token) || 0) + xScale.bandwidth() / 2)
      .attr('y', (d) => yScale(d.probability) - 5)
      .attr('text-anchor', 'middle')
      .attr('class', (d, i) => `text-xs font-semibold ${i === 0 ? 'fill-amber-600' : 'fill-gray-700'}`)
      .text((d) => `${(d.probability * 100).toFixed(1)}%`);

    g.selectAll('.token-label')
      .data(filteredPredictions)
      .join('text')
      .attr('class', 'token-label')
      .attr('x', (d) => (xScale(d.token) || 0) + xScale.bandwidth() / 2)
      .attr('y', height - margin.top - margin.bottom + 20)
      .attr('text-anchor', 'end')
      .attr('transform', (d) => {
        const x = (xScale(d.token) || 0) + xScale.bandwidth() / 2;
        const y = height - margin.top - margin.bottom + 20;
        return `rotate(-45, ${x}, ${y})`;
      })
      .attr('class', 'text-xs fill-gray-700')
      .text((d) => d.token);

    g.append('g')
      .attr('transform', `translate(0,${height - margin.top - margin.bottom})`)
      .call(d3.axisBottom(xScale).tickSize(0).tickFormat(() => ''))
      .select('.domain')
      .attr('stroke', '#9ca3af');

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => `${(Number(d) * 100).toFixed(0)}%`))
      .attr('class', 'text-xs')
      .selectAll('text')
      .attr('class', 'fill-gray-600');
  }, [filteredPredictions]);

  return (
    <div>
      {topPrediction && (
        <div className="mb-6 p-6 bg-gradient-to-r from-amber-50 to-yellow-50 border-2 border-amber-400 rounded-lg shadow-lg">
          <div className="flex items-center justify-center space-x-4">
            <span className="text-3xl">✓</span>
            <div>
              <div className="text-lg font-bold text-gray-800">
                预测结果: <span className="text-amber-600 text-2xl">{topPrediction.token}</span>
              </div>
              <div className="text-sm text-gray-600">
                概率: <span className="font-semibold">{(topPrediction.probability * 100).toFixed(2)}%</span>
              </div>
            </div>
          </div>
        </div>
      )}
      <p className="text-gray-700 mb-4">最终预测结果 - 最高概率的token被标记为预测输出</p>
      <svg ref={svgRef}></svg>
    </div>
  );
};

const TopKPredictions: React.FC<{ predictions: LogitData[] }> = ({ predictions }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h4 className="text-lg font-bold text-gray-800 mb-4">Top-10 预测候选</h4>
      <div className="space-y-2">
        {predictions.slice(0, 10).map((pred, idx) => (
          <div
            key={pred.tokenId}
            className={`flex items-center justify-between p-3 rounded-lg ${
              idx === 0 ? 'bg-amber-50 border-2 border-amber-400' : 'bg-gray-50'
            }`}
          >
            <div className="flex items-center space-x-3">
              <span className={`font-bold ${idx === 0 ? 'text-amber-600' : 'text-gray-600'}`}>{idx + 1}.</span>
              <span className={`font-semibold ${idx === 0 ? 'text-amber-800 text-lg' : 'text-gray-800'}`}>
                {pred.token}
              </span>
              {idx === 0 && <span className="text-xl">✓</span>}
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-48 bg-gray-200 rounded-full h-2.5">
                <div
                  className={`${idx === 0 ? 'bg-amber-500' : 'bg-blue-500'} h-2.5 rounded-full transition-all`}
                  style={{ width: `${pred.probability * 100}%` }}
                ></div>
              </div>
              <span className={`font-semibold ${idx === 0 ? 'text-amber-600' : 'text-gray-700'}`}>
                {(pred.probability * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
