import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { I18nextProvider } from 'react-i18next'
import { describe, expect, it } from 'vitest'
import App from '../App'
import { ThemeProvider } from '../hooks/useTheme'
import i18n from '../services/i18n'

const renderWithProviders = () =>
  render(
    <I18nextProvider i18n={i18n}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </I18nextProvider>,
  )

describe('可视化交互', () => {
  it('支持逐步执行状态切换', async () => {
    renderWithProviders()
    const user = userEvent.setup()

    expect(screen.getByText(/步骤 1\/\d/)).toBeInTheDocument()

    const nextButton = screen.getByRole('button', { name: '下一步' })
    await user.click(nextButton)

    expect(screen.getByText(/步骤 2\/\d/)).toBeInTheDocument()

    const statusElements = screen.getAllByRole('status')
    expect(statusElements.some((element) => element.textContent?.includes('第 2 步 ·'))).toBe(true)
  })

  it('点击层级时更新右侧详情并渲染图表', async () => {
    renderWithProviders()
    const user = userEvent.setup()

    const stepButton = screen.getByRole('button', { name: '第 2 步 · 注意力推理' })
    await user.click(stepButton)

    const layerButton = await screen.findByRole('button', { name: '多头注意力' })
    await user.click(layerButton)

    expect(await screen.findByText('当前层：多头注意力')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /稀疏注意力矩阵 · 多头注意力/ })).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /MoE 路由流向图 · 多头注意力/ })).toBeInTheDocument()
  })
})
