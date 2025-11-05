import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { useTheme } from '../hooks/useTheme'

const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme()
  const { t } = useTranslation()

  const label = useMemo(
    () => (theme === 'dark' ? t('actions.switchToLight') : t('actions.switchToDark')),
    [t, theme],
  )

  return (
    <button
      type="button"
      onClick={toggleTheme}
      className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-surface px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-primary hover:text-primary dark:border-slate-700 dark:bg-surface-dark dark:text-slate-200"
    >
      <span className="text-lg" aria-hidden>
        {theme === 'dark' ? '🌙' : '☀️'}
      </span>
      <span>{label}</span>
      <span className="sr-only">{t('layout.themeToggle')}</span>
    </button>
  )
}

export default ThemeToggle
