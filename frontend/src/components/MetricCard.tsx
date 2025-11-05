type MetricCardProps = {
  title: string
  value: string
  trend: string
  trendLabel: string
  highlight?: 'up' | 'down'
}

const MetricCard = ({ title, value, trend, trendLabel, highlight = 'up' }: MetricCardProps) => {
  const trendColor = highlight === 'up' ? 'text-emerald-500' : 'text-rose-500'

  return (
    <article className="card">
      <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{title}</p>
      <p className="mt-3 text-3xl font-semibold text-slate-900 dark:text-slate-50">{value}</p>
      <p className="mt-4 flex items-baseline gap-2 text-sm text-slate-500 dark:text-slate-400">
        <span className={trendColor}>{trend}</span>
        <span>{trendLabel}</span>
      </p>
    </article>
  )
}

export default MetricCard
