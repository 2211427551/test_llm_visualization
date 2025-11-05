import {
  PropsWithChildren,
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react'

type Theme = 'light' | 'dark'

type ThemeContextValue = {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

const THEME_STORAGE_KEY = 'dashboard-preferred-theme'

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined)

const getInitialTheme = (): Theme => {
  if (typeof window === 'undefined') {
    return 'light'
  }

  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
  if (storedTheme === 'light' || storedTheme === 'dark') {
    return storedTheme
  }

  const prefersDark = window.matchMedia
    ? window.matchMedia('(prefers-color-scheme: dark)').matches
    : false

  return prefersDark ? 'dark' : 'light'
}

export const ThemeProvider = ({ children }: PropsWithChildren) => {
  const [theme, setTheme] = useState<Theme>(() => getInitialTheme())

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    const root = window.document.documentElement
    root.classList.toggle('dark', theme === 'dark')
    root.dataset.theme = theme
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      toggleTheme: () => setTheme((current) => (current === 'dark' ? 'light' : 'dark')),
    }),
    [theme],
  )

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export const useTheme = () => {
  const context = useContext(ThemeContext)

  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }

  return context
}
