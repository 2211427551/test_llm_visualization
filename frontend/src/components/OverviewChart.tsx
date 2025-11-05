import { useEffect, useMemo, useRef } from 'react'
import * as d3 from 'd3'
import { useTranslation } from 'react-i18next'
import { useTheme } from '../hooks/useTheme'

type DataPoint = {
  label: string
  value: number
}

const CHART_WIDTH = 360
const CHART_HEIGHT = 220
const MARGINS = { top: 20, right: 24, bottom: 32, left: 44 }

const OverviewChart = () => {
  const svgRef = useRef<SVGSVGElement | null>(null)
  const { t } = useTranslation()
  const { theme } = useTheme()

  const data = useMemo<DataPoint[]>(
    () => [
      { label: '周一', value: 120 },
      { label: '周二', value: 98 },
      { label: '周三', value: 156 },
      { label: '周四', value: 132 },
      { label: '周五', value: 180 },
      { label: '周六', value: 210 },
      { label: '周日', value: 164 },
    ],
    [],
  )

  useEffect(() => {
    if (!svgRef.current) {
      return
    }

    const isDark = theme === 'dark'
    const accentStart = isDark ? '#60a5fa' : '#2563eb'
    const accentEnd = isDark ? '#1d4ed8' : '#93c5fd'
    const textColor = isDark ? '#cbd5f5' : '#64748b'
    const gridColor = isDark ? '#1e293b' : '#e2e8f0'

    const svg = d3.select(svgRef.current)
    const width = CHART_WIDTH - MARGINS.left - MARGINS.right
    const height = CHART_HEIGHT - MARGINS.top - MARGINS.bottom

    svg.selectAll('*').remove()

    svg.attr('viewBox', `0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`)

    const defs = svg.append('defs')
    const gradient = defs
      .append('linearGradient')
      .attr('id', 'overview-gradient')
      .attr('x1', '0%')
      .attr('x2', '0%')
      .attr('y1', '0%')
      .attr('y2', '100%')

    gradient
      .append('stop')
      .attr('offset', '0%')
      .attr('stop-color', accentStart)
      .attr('stop-opacity', 0.9)

    gradient
      .append('stop')
      .attr('offset', '100%')
      .attr('stop-color', accentEnd)
      .attr('stop-opacity', 0.2)

    const container = svg
      .append('g')
      .attr('transform', `translate(${MARGINS.left},${MARGINS.top})`)

    const x = d3
      .scaleBand<string>()
      .domain(data.map((d) => d.label))
      .range([0, width])
      .padding(0.35)

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(data, (d) => d.value)! * 1.1])
      .nice()
      .range([height, 0])

    const grid = container
      .append('g')
      .attr('class', 'grid')
      .call(
        d3
          .axisLeft(y)
          .ticks(4)
          .tickSize(-width)
          .tickFormat(() => ''),
      )

    grid
      .selectAll('line')
      .attr('stroke', gridColor)
      .attr('stroke-dasharray', '4,8')

    grid.selectAll('path').attr('stroke', 'transparent')

    container
      .append('g')
      .attr('fill', 'url(#overview-gradient)')
      .selectAll('rect')
      .data(data)
      .join('rect')
      .attr('x', (d) => x(d.label) ?? 0)
      .attr('y', (d) => y(d.value))
      .attr('width', x.bandwidth())
      .attr('height', (d) => height - y(d.value))
      .attr('rx', 10)

    const xAxis = container
      .append('g')
      .attr('transform', `translate(0, ${height})`)
      .call(d3.axisBottom(x).tickSize(0))

    xAxis
      .selectAll('text')
      .attr('fill', textColor)
      .attr('font-size', 12)
      .style('font-family', 'Noto Sans SC, Inter, sans-serif')

    xAxis.select('.domain').attr('stroke', 'transparent')

    const yAxis = container.append('g').call(d3.axisLeft(y).ticks(4).tickSize(0))

    yAxis
      .selectAll('text')
      .attr('fill', textColor)
      .attr('font-size', 12)
      .style('font-family', 'Noto Sans SC, Inter, sans-serif')

    yAxis.select('.domain').attr('stroke', 'transparent')
  }, [data, theme])

  return (
    <section className="card h-full">
      <header className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-slate-500 dark:text-slate-400">
            {t('dashboard.highlight')}
          </p>
          <h2 className="mt-1 text-lg font-semibold text-slate-900 dark:text-slate-50">
            {t('dashboard.chartTitle')}
          </h2>
        </div>
      </header>
      <svg
        ref={svgRef}
        role="img"
        aria-label={t('dashboard.chartTitle')}
        className="mt-6 h-52 w-full"
      />
    </section>
  )
}

export default OverviewChart
