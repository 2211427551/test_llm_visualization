import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import MetricCard from './components/MetricCard'
import OverviewChart from './components/OverviewChart'
import MainLayout from './layouts/MainLayout'

type MetricKey = 'users' | 'retention' | 'conversion'

type MetricDefinition = {
  key: MetricKey
  value: string
  trend: string
  highlight: 'up' | 'down'
}

const App = () => {
  const { t } = useTranslation()

  const metrics = useMemo<MetricDefinition[]>(
    () => [
      { key: 'users', value: '1,240', trend: '+8.3%', highlight: 'up' },
      { key: 'retention', value: '62.4%', trend: '+2.1%', highlight: 'up' },
      { key: 'conversion', value: '12.6%', trend: '-0.6%', highlight: 'down' },
    ],
    [],
  )

  const tips = t('dashboard.tips', { returnObjects: true }) as string[]

  return (
    <MainLayout>
      <section className="grid gap-6 md:grid-cols-3">
        {metrics.map((metric) => (
          <MetricCard
            key={metric.key}
            title={t(`dashboard.metrics.${metric.key}`)}
            value={metric.value}
            trend={metric.trend}
            trendLabel={t('dashboard.trendHint')}
            highlight={metric.highlight}
          />
        ))}
      </section>

      <section className="mt-8 grid gap-6 lg:grid-cols-[2fr_1fr]">
        <OverviewChart />
        <aside className="card flex flex-col justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">
              {t('dashboard.highlight')}
            </h2>
            <p className="mt-3 text-sm leading-6 text-slate-600 dark:text-slate-300">
              {t('dashboard.callout')}
            </p>
            <ul className="mt-4 space-y-2 text-sm text-slate-600 dark:text-slate-300">
              {tips.map((tip) => (
                <li key={tip} className="flex items-start gap-2">
                  <span className="mt-1 inline-block h-1.5 w-1.5 rounded-full bg-primary" aria-hidden />
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>
          <div className="mt-6 rounded-2xl bg-gradient-to-r from-primary to-secondary px-4 py-3 text-sm font-medium text-primary-contrast shadow-soft-lg">
            {t('dashboard.promo')}
          </div>
        </aside>
      </section>
    </MainLayout>
  )
}

export default App
