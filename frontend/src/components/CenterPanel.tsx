import { useEffect, useMemo, useRef } from 'react'
import { easeCubicInOut, scalePoint, select } from 'd3'
import { useVisualizationState } from '../hooks/useVisualizationState'
import type { LayerType } from '../types/visualization'

const layerColors: Record<LayerType, { fill: string; stroke: string }> = {
  input: { fill: '#bfdbfe', stroke: '#2563eb' },
  embedding: { fill: '#bbf7d0', stroke: '#16a34a' },
  attention: { fill: '#e9d5ff', stroke: '#7c3aed' },
  feedforward: { fill: '#fed7aa', stroke: '#f97316' },
  output: { fill: '#fecaca', stroke: '#ef4444' },
}

const CenterPanel = () => {
  const { currentStep, selectedLayerId, selectLayer, isLayerSelected } = useVisualizationState()
  const svgRef = useRef<SVGSVGElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  const layers = currentStep.layers
  const hasLayers = layers.length > 0

  const chartHeight = useMemo(() => Math.max(layers.length * 110, 360), [layers.length])

  useEffect(() => {
    if (!svgRef.current) {
      return
    }

    if (!hasLayers) {
      select(svgRef.current).selectAll('*').remove()
      return
    }

    const width = containerRef.current?.clientWidth ?? 360
    const height = chartHeight
    const padding = 48
    const xCenter = width / 2

    const svg = select(svgRef.current)
    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const yScale = scalePoint<string>()
      .domain(layers.map((layer) => layer.id))
      .range([padding, height - padding])
      .padding(0.5)

    const linkData = layers.slice(1).map((layer, index) => ({
      id: `${layers[index].id}-${layer.id}`,
      source: layers[index],
      target: layer,
    }))

    const pathForLink = (sourceId: string, targetId: string) => {
      const sourceY = yScale(sourceId) ?? padding
      const targetY = yScale(targetId) ?? height - padding
      const xOffset = 56
      return `M ${xCenter - xOffset} ${sourceY} C ${xCenter - xOffset / 2} ${sourceY}, ${xCenter + xOffset / 2} ${targetY}, ${xCenter + xOffset} ${targetY}`
    }

    const links = svg.selectAll<SVGPathElement, typeof linkData[number]>('path.layer-link')

    links
      .data(linkData, (d) => d.id)
      .join(
        (enter) =>
          enter
            .append('path')
            .attr('class', 'layer-link')
            .attr('fill', 'none')
            .attr('stroke', '#94a3b8')
            .attr('stroke-width', 1.5)
            .attr('opacity', 0)
            .attr('d', (d) => pathForLink(d.source.id, d.target.id))
            .transition()
            .duration(600)
            .ease(easeCubicInOut)
            .attr('opacity', 0.35),
        (update) =>
          update
            .transition()
            .duration(600)
            .ease(easeCubicInOut)
            .attr('opacity', 0.35)
            .attr('d', (d) => pathForLink(d.source.id, d.target.id)),
        (exit) => exit.transition().duration(300).attr('opacity', 0).remove(),
      )

    const nodes = svg.selectAll<SVGGElement, (typeof layers)[number]>('g.layer-node')

    const nodeEnter = nodes
      .data(layers, (d) => d.id)
      .join((enter) => {
        const group = enter
          .append('g')
          .attr('class', 'layer-node')
          .attr('tabindex', 0)
          .attr('role', 'button')
          .attr('aria-label', (d) => `选择层 ${d.name}`)
          .style('cursor', 'pointer')
          .attr('transform', (d) => `translate(${xCenter}, ${yScale(d.id) ?? padding})`)
          .attr('opacity', 0)

        group
          .append('circle')
          .attr('r', 34)
          .attr('fill', (d) => layerColors[d.type].fill)
          .attr('stroke', (d) => (isLayerSelected(d.id) ? layerColors[d.type].stroke : '#cbd5f5'))
          .attr('stroke-width', (d) => (isLayerSelected(d.id) ? 4 : 2))

        group
          .append('text')
          .attr('class', 'layer-label')
          .attr('text-anchor', 'middle')
          .attr('dominant-baseline', 'middle')
          .attr('fill', '#0f172a')
          .attr('font-size', 12)
          .text((d) => d.name)

        group.append('title').text((d) => d.summary)

        return group
      })

    nodeEnter
      .transition()
      .duration(600)
      .ease(easeCubicInOut)
      .attr('opacity', 1)

    const mergedNodes = svg.selectAll<SVGGElement, (typeof layers)[number]>('g.layer-node')

    mergedNodes
      .transition()
      .duration(500)
      .ease(easeCubicInOut)
      .attr('transform', (d) => `translate(${xCenter}, ${yScale(d.id) ?? padding})`)

    mergedNodes
      .select<SVGCircleElement>('circle')
      .transition()
      .duration(500)
      .ease(easeCubicInOut)
      .attr('fill', (d) => layerColors[d.type].fill)
      .attr('stroke', (d) => (isLayerSelected(d.id) ? layerColors[d.type].stroke : '#cbd5f5'))
      .attr('stroke-width', (d) => (isLayerSelected(d.id) ? 4 : 2))
      .attr('opacity', (d) => (isLayerSelected(d.id) ? 1 : 0.9))

    mergedNodes
      .select<SVGTextElement>('text.layer-label')
      .transition()
      .duration(400)
      .attr('fill', (d) => (isLayerSelected(d.id) ? '#0f172a' : '#1e293b'))

    mergedNodes.select('title').text((d) => d.summary)

    mergedNodes.on('click', (_, datum) => {
      selectLayer(datum.id)
    })

    mergedNodes.on('keydown', (event, datum) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        selectLayer(datum.id)
      }
    })
  }, [chartHeight, hasLayers, isLayerSelected, layers, selectLayer])

  const selectedLayer = useMemo(() => layers.find((layer) => layer.id === selectedLayerId) ?? layers[0], [
    layers,
    selectedLayerId,
  ])

  return (
    <div className="flex h-full flex-col">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">模型结构视图</h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          点击或按下回车选择 Transformer 层，右侧将展示对应视图。
        </p>
      </div>

      <div className="flex-1 rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div ref={containerRef} className="relative h-full" aria-hidden="false">
          <svg
            ref={svgRef}
            role="img"
            aria-labelledby="model-structure-title model-structure-desc"
            className="h-full w-full"
          >
            <title id="model-structure-title">Transformer 模型结构</title>
            <desc id="model-structure-desc">
              展示当前步骤内各层的顺序与连接，可通过键盘空格与回车选择。
            </desc>
          </svg>
          {!hasLayers && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-slate-400 dark:text-slate-500">
              暂无可视化数据，请先执行推理。
            </div>
          )}
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-2" aria-label="可选层列表">
          {layers.map((layer) => {
            const active = isLayerSelected(layer.id)
            return (
              <button
                key={layer.id}
                type="button"
                onClick={() => selectLayer(layer.id)}
                className={`rounded-full border px-3 py-1 text-xs transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 dark:border-slate-700 ${
                  active
                    ? 'border-primary bg-primary/10 text-primary dark:text-primary-contrast'
                    : 'border-slate-200 text-slate-600 hover:border-primary/40 hover:text-primary dark:text-slate-300'
                }`}
                aria-pressed={active}
              >
                {layer.name}
              </button>
            )
          })}
        </div>
      </div>

      <div className="mt-4 rounded-lg bg-blue-50 p-3 text-sm text-blue-800 dark:bg-blue-900/30 dark:text-blue-200">
        <strong>已选层：</strong> {selectedLayer?.name} · {selectedLayer?.summary}
      </div>

      <div className="mt-3 rounded-lg bg-white p-3 text-xs text-slate-600 shadow-sm dark:bg-slate-800 dark:text-slate-300">
        <div className="font-medium">图例</div>
        <div className="mt-2 grid grid-cols-2 gap-2">
          {(
            [
              { type: 'input', label: '输入层' },
              { type: 'embedding', label: '嵌入层' },
              { type: 'attention', label: '注意力层' },
              { type: 'feedforward', label: '前馈层' },
              { type: 'output', label: '输出层' },
            ] as { type: LayerType; label: string }[]
          ).map((item) => (
            <div key={item.type} className="flex items-center gap-2">
              <span
                className="inline-block h-3 w-3 rounded-full border"
                style={{
                  backgroundColor: layerColors[item.type].fill,
                  borderColor: layerColors[item.type].stroke,
                }}
                aria-hidden="true"
              />
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default CenterPanel
