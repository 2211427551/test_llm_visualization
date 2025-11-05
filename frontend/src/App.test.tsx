import { render, screen } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import { describe, expect, it } from 'vitest'
import App from './App'
import { ThemeProvider } from './hooks/useTheme'
import i18n from './services/i18n'

describe('App', () => {
  it('renders localized content and chart title', () => {
    render(
      <I18nextProvider i18n={i18n}>
        <ThemeProvider>
          <App />
        </ThemeProvider>
      </I18nextProvider>,
    )

    expect(screen.getByText(/欢迎回来/)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /七日访问量趋势/ })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Transformer 模型结构/ })).toBeInTheDocument()
  })
})
