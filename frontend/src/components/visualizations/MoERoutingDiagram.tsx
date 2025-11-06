import { useEffect, useMemo, useRef } from 'react'
import { easeCubicInOut, scaleLinear, scalePoint, select } from 'd3'
import { limitMoERouting } from '../../utils/dataTransform'
import type { MoERoutingData } from '../../types/visualization'

interface MoERoutingDiagramProps {
  data: MoERoutingData
  ariaLabel?: string
}

type RouteDatum = MoERoutingData['routes'][number]

type TokenDatum = MoERoutingData['tokens'][number]

type ExpertDatum = MoERoutingData['experts'][number]

const MoERoutingDiagram = ({ data, ariaLabel = 'MoE 路由流向图' }: MoERoutingDiagramProps) => {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const processedData = useMemo(() => limitMoERouting(data), [data])

  const helpers = useMemo(() => {
    const tokenMap = new Map(processedData.tokens.map((token) => [token.id, token.label]))
    const expertMap = new Map(processedData.experts.map((expert) => [expert.id, expert.label]))
    return { tokenMap, expertMap }
  }, [processedData.experts, processedData.tokens])

  useEffect(() => {
    if (!svgRef.current) {
      return
    }

    const tokenCount = processedData.tokens.length
    const expertCount = processedData.experts.length
    const height = Math.max(tokenCount, expertCount) * 64 + 80
    const width = 420
    const leftX = 80
    const rightX = width - 80

    const svg = select(svgRef.current)
    svg.attr('viewBox', `0 0 ${width} ${height}`)

    const tokenScale = scalePoint<string>()
      .domain(processedData.tokens.map((token) => token.id))
      .range([40, height - 40])
      .padding(0.5)

    const expertScale = scalePoint<string>()
      .domain(processedData.experts.map((expert) => expert.id))
      .range([40, height - 40])
      .padding(0.5)

    const maxWeight = processedData.routes.reduce((acc, route) => Math.max(acc, route.weight), 0)
    const weightScale = scaleLinear().domain([0, maxWeight || 1]).range([1, 14])

    const createRoutePath = (route: RouteDatum) => {
      const sourceY = tokenScale(route.tokenId) ?? 40
      const targetY = expertScale(route.expertId) ?? height - 40
      const midX = (leftX + rightX) / 2
      return `M ${leftX} ${sourceY} C ${midX} ${sourceY}, ${midX} ${targetY}, ${rightX} ${targetY}`
    }

    const routes = svg.selectAll<SVGPathElement, RouteDatum>('path.route')

    const mergedRoutes = routes
      .data(processedData.routes, (route) => `${route.tokenId}-${route.expertId}`)
      .join(
        (enter) =>
          enter
            .append('path')
            .attr('class', 'route')
            .attr('fill', 'none')
            .attr('stroke', '#38bdf8')
            .attr('stroke-linecap', 'round')
            .attr('stroke-opacity', 0)
            .attr('d', (route) => createRoutePath(route))
            .call((selection) => selection.append('title')),
        (update) => update,
        (exit) => exit.transition().duration(300).attr('stroke-opacity', 0).remove(),
      )

    mergedRoutes
      .transition()
      .duration(600)
      .ease(easeCubicInOut)
      .attr('stroke-width', (route) => weightScale(route.weight))
      .attr('stroke-opacity', (route) => 0.25 + (route.weight / (maxWeight || 1)) * 0.65)
      .attr('d', (route) => createRoutePath(route))

    mergedRoutes
      .select('title')
      .text((route) => {
        const tokenLabel = helpers.tokenMap.get(route.tokenId) ?? route.tokenId
        const expertLabel = helpers.expertMap.get(route.expertId) ?? route.expertId
        return `${tokenLabel} → ${expertLabel} · 权重 ${(route.weight * 100).toFixed(1)}%`
      })

    const renderNodes = <T extends TokenDatum | ExpertDatum>(
      selector: string,
      items: T[],
      xPosition: number,
      yScale: (value: string) => number | undefined,
      fillColor: string,
    ) => {
      const className = selector.includes('.') ? selector.split('.').pop() ?? '' : selector
      const groups = svg.selectAll<SVGGElement, T>(selector)

      const mergedGroups = groups
        .data(items, (item) => item.id)
        .join((enter) => {
          const group = enter
            .append('g')
            .attr('class', className)
            .attr('transform', (item) => {
              const y = yScale(item.id) ?? 40
              return `translate(${xPosition}, ${y})`
            })
            .attr('opacity', 0)

          group
            .append('rect')
            .attr('x', -48)
            .attr('y', -16)
            .attr('width', 96)
            .attr('height', 32)
            .attr('rx', 16)
            .attr('ry', 16)
            .attr('fill', fillColor)
            .attr('stroke', '#0ea5e9')
            .attr('stroke-width', 1)

          group
            .append('text')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('font-size', 11)
            .attr('fill', '#0f172a')
            .text((item) => ('label' in item ? item.label : item.id))

          group.append('title').text((item) => ('label' in item ? item.label : item.id))

          return group
        })

      mergedGroups
        .transition()
        .duration(500)
        .ease(easeCubicInOut)
        .attr('opacity', 1)
        .attr('transform', (item) => {
          const y = yScale(item.id) ?? 40
          return `translate(${xPosition}, ${y})`
        })

      mergedGroups.select('title').text((item) => ('label' in item ? item.label : item.id))
    }

    renderNodes<TokenDatum>('g.token-node', processedData.tokens, leftX, tokenScale, '#bae6fd')
    renderNodes<ExpertDatum>('g.expert-node', processedData.experts, rightX, expertScale, '#c4b5fd')
  }, [helpers, processedData])

  return <svg ref={svgRef} role="img" aria-label={ariaLabel} className="h-auto w-full" focusable="false" />
}

export default MoERoutingDiagram
