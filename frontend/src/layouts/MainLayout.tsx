import { PropsWithChildren } from 'react'
import { useTranslation } from 'react-i18next'
import ThemeToggle from '../components/ThemeToggle'

const MainLayout = ({ children }: PropsWithChildren) => {
  const { t } = useTranslation()

  return (
    <div className="min-h-screen bg-background transition-colors duration-300 dark:bg-background-dark">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-10 md:px-8">
        <header className="flex flex-col items-start justify-between gap-6 pb-10 md:flex-row md:items-center">
          <div>
            <p className="text-sm font-medium text-primary dark:text-primary-contrast">
              {t('dashboard.welcome')}
            </p>
            <h1 className="mt-2 text-3xl md:text-4xl">{t('layout.title')}</h1>
            <p className="mt-3 max-w-2xl text-base text-slate-600 dark:text-slate-300">
              {t('layout.subtitle')}
            </p>
          </div>
          <ThemeToggle />
        </header>

        <main className="flex-1">{children}</main>

        <footer className="pt-10 text-sm text-slate-500 dark:text-slate-400">
          {t('layout.footer')}
        </footer>
      </div>
    </div>
  )
}

export default MainLayout
