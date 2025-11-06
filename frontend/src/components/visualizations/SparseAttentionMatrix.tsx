import { useEffect, useMemo, useRef } from 'react'
import { interpolatePurples, scaleLinear, select } from 'd3'
import { downsampleSparseMatrix } from '../../utils/dataTransform'
import type { SparseAttentionData } from '../../types/visualization'

interface SparseAttentionMatrixProps {
  data: SparseAttentionData
  ariaLabel?: string
}

interface AttentionCell {
  rowIndex: number
  columnIndex: number
  value: number
  isSparse: boolean
}

const SparseAttentionMatrix = ({ data, ariaLabel = '稀疏注意力矩阵' }: SparseAttentionMatrixProps) => {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const processedMatrix = useMemo(() => downsampleSparseMatrix(data.matrix), [data.matrix])
  const processedLabels = useMemo(() => {
    if (processedMatrix.length === 0) {
      return [] as string[]
    }

    if (data.headLabels.length === processedMatrix.length) {
      return data.headLabels
    }

    const step = Math.max(1, Math.floor(data.headLabels.length / processedMatrix.length))
    return processedMatrix.map((_, index) => {
      const labelIndex = Math.min(index * step, data.headLabels.length - 1)
      return data.headLabels[labelIndex] ?? `位置 ${index + 1}`
    })
  }, [data.headLabels, processedMatrix])

  useEffect(() => {
    if (!svgRef.current || processedMatrix.length === 0 || processedMatrix[0].length === 0) {
      if (svgRef.current) {
        select(svgRef.current).selectAll('*').remove()
      }
      return
    }

    const size = processedMatrix.length
    const cellSize = Math.max(Math.min(28, 220 / size), 14)
    const labelPadding = 36
    const width = size * cellSize + labelPadding + 8
    const height = size * cellSize + labelPadding + 8

    const svg = select(svgRef.current)
    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const intensityScale = scaleLinear().domain([0, 1]).range([0.15, 0.95])

    const flattened: AttentionCell[] = processedMatrix.flatMap((row, rowIndex) =>
      row.map((cell, columnIndex) => ({
        rowIndex,
        columnIndex,
        value: cell.value,
        isSparse: cell.isSparse,
      })),
    )

    const cells = svg
      .selectAll<SVGRectElement, AttentionCell>('rect.attention-cell')
      .data(flattened, (d) => `${d.rowIndex}-${d.columnIndex}`)

    const joined = cells
      .join(
        (enter) =>
          enter
            .append('rect')
            .attr('class', 'attention-cell')
            .attr('x', (d) => labelPadding + d.columnIndex * cellSize)
            .attr('y', (d) => 8 + d.rowIndex * cellSize)
            .attr('width', cellSize - 1)
            .attr('height', cellSize - 1)
            .attr('rx', 2)
            .attr('ry', 2)
            .attr('stroke-width', 1)
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
      .attr('fill', (d) => (d.isSparse ? '#e2e8f0' : interpolatePurples(intensityScale(d.value))))
      .attr('stroke', (d) => (d.isSparse ? '#cbd5f5' : '#7c3aed'))
      .attr('stroke-dasharray', (d) => (d.isSparse ? '3,2' : ''))

    joined
      .select('title')
      .text((d) =>
        `${d.isSparse ? '稀疏区域' : '活跃连接'} · 位置 [${d.rowIndex + 1}, ${d.columnIndex + 1}] · 权重 ${d.value.toFixed(3)}`,
      )

    const columnLabels = svg.selectAll<SVGTextElement, string>('text.column-label')

    columnLabels
      .data(processedLabels)
      .join(
        (enter) =>
          enter
            .append('text')
            .attr('class', 'column-label')
            .attr('text-anchor', 'middle')
            .attr('font-size', 10)
            .attr('fill', '#475569')
            .attr('x', (_, index) => labelPadding + index * cellSize + cellSize / 2)
            .attr('y', labelPadding - 8)
            .attr('opacity', 0)
            .text((label) => label),
        (update) => update,
        (exit) => exit.remove(),
      )
      .transition()
      .duration(400)
      .attr('opacity', 1)

    const rowLabels = svg.selectAll<SVGTextElement, string>('text.row-label')

    rowLabels
      .data(processedLabels)
      .join(
        (enter) =>
          enter
            .append('text')
            .attr('class', 'row-label')
            .attr('text-anchor', 'end')
            .attr('font-size', 10)
            .attr('fill', '#475569')
            .attr('x', labelPadding - 6)
            .attr('y', (_, index) => 8 + index * cellSize + cellSize / 2 + 3)
            .attr('opacity', 0)
            .text((label) => label),
        (update) => update,
        (exit) => exit.remove(),
      )
      .transition()
      .duration(400)
      .attr('opacity', 1)
  }, [processedLabels, processedMatrix])

  return <svg ref={svgRef} role="img" aria-label={ariaLabel} className="h-auto w-full" focusable="false" />
}

export default SparseAttentionMatrix
