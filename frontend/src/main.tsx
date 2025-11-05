import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { I18nextProvider } from 'react-i18next'
import App from './App.tsx'
import { ThemeProvider } from './hooks/useTheme'
import i18n from './services/i18n'
import './styles/index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <I18nextProvider i18n={i18n}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </I18nextProvider>
  </StrictMode>,
)
