import { render, screen, waitFor } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import { describe, expect, it, vi } from 'vitest'
import App from './App'
import { ThemeProvider } from './hooks/useTheme'
import i18n from './services/i18n'

const mockInitializeResponse = {
  success: true,
  message: '模型初始化成功',
  config: {
    modelName: 'Transformer-MoE',
    vocabSize: 32000,
    contextSize: 256,
    nLayer: 6,
    nHead: 8,
    nEmbed: 512,
    dropout: 0.1,
    useSparseAttention: true,
    useMoe: true,
    moeNumExperts: 4,
    moeTopK: 2,
    initializedAt: '2025-01-01T00:00:00Z',
  },
}

describe('App', () => {
  it('renders localized content and chart title', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        new Response(JSON.stringify(mockInitializeResponse), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )

    render(
      <I18nextProvider i18n={i18n}>
        <ThemeProvider>
          <App />
        </ThemeProvider>
      </I18nextProvider>,
    )

    expect(screen.getByText(/欢迎回来/)).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /七日访问量趋势/ })).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByRole('img', { name: /Transformer 模型结构/ })).toBeInTheDocument()
    })
  })
})
