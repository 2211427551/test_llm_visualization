import { useEffect, useMemo, useRef } from 'react'
import { interpolateBlues, scaleLinear, select } from 'd3'
import { downsampleMatrix } from '../../utils/dataTransform'

interface TensorHeatmapProps {
  data: number[][]
  ariaLabel?: string
}

interface HeatmapCell {
  rowIndex: number
  columnIndex: number
  value: number
}

const TensorHeatmap = ({ data, ariaLabel = '张量热力图' }: TensorHeatmapProps) => {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const processedData = useMemo(() => downsampleMatrix(data), [data])

  useEffect(() => {
    if (!svgRef.current || processedData.length === 0 || processedData[0].length === 0) {
      if (svgRef.current) {
        select(svgRef.current).selectAll('*').remove()
      }
      return
    }

    const rows = processedData.length
    const cols = processedData[0].length
    const cellSize = Math.max(Math.min(30, 240 / Math.max(rows, cols)), 16)
    const width = cols * cellSize
    const height = rows * cellSize

    const svg = select(svgRef.current)
    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const intensityScale = scaleLinear().domain([0, 1]).range([0.1, 0.95])

    const flattened: HeatmapCell[] = processedData.flatMap((row, rowIndex) =>
      row.map((value, columnIndex) => ({ rowIndex, columnIndex, value })),
    )

    const cells = svg
      .selectAll<SVGRectElement, HeatmapCell>('rect.heatmap-cell')
      .data(flattened, (d) => `${d.rowIndex}-${d.columnIndex}`)

    const joined = cells.join(
      (enter) =>
        enter
          .append('rect')
          .attr('class', 'heatmap-cell')
          .attr('x', (d) => d.columnIndex * cellSize)
          .attr('y', (d) => d.rowIndex * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('rx', 3)
          .attr('ry', 3)
          .attr('stroke', '#e2e8f0')
          .attr('stroke-width', 0.5)
          .attr('opacity', 0)
          .call((selection) => {
            selection.append('title')
          }),
      (update) => update,
      (exit) => exit.transition().duration(200).attr('opacity', 0).remove(),
    )

    joined
      .transition()
      .duration(500)
      .attr('opacity', 1)
      .attr('fill', (d) => interpolateBlues(intensityScale(d.value)))

    joined
      .select('title')
      .text((d) => `位置 [${d.rowIndex + 1}, ${d.columnIndex + 1}] · 值 ${d.value.toFixed(3)}`)
  }, [processedData])

  return (
    <svg
      ref={svgRef}
      role="img"
      aria-label={ariaLabel}
      className="h-auto w-full"
      focusable="false"
    />
  )
}

export default TensorHeatmap
