import type { PropsWithChildren } from 'react'
import { useTranslation } from 'react-i18next'
import ThemeToggle from '../components/ThemeToggle'

const MainLayout = ({ children }: PropsWithChildren) => {
  const { t } = useTranslation()

  return (
    <div className="min-h-screen bg-background transition-colors duration-300 dark:bg-background-dark">
      <div className="mx-auto flex min-h-screen max-w-full flex-col px-4 py-6 md:px-6 lg:px-8">
        <header className="flex flex-col items-start justify-between gap-4 pb-6 md:flex-row md:items-center">
          <div>
            <p className="text-sm font-medium text-primary dark:text-primary-contrast">
              {t('dashboard.welcome')}
            </p>
            <h1 className="mt-2 text-2xl md:text-3xl">{t('layout.title')}</h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-600 dark:text-slate-300">
              {t('layout.subtitle')}
            </p>
          </div>
          <ThemeToggle />
        </header>

        <main className="flex-1 overflow-hidden">{children}</main>

        <footer className="pt-6 text-sm text-slate-500 dark:text-slate-400">
          {t('layout.footer')}
        </footer>
      </div>
    </div>
  )
}

export default MainLayout
