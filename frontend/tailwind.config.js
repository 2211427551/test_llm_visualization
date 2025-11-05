/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2563eb',
          contrast: '#f8fafc',
        },
        secondary: {
          DEFAULT: '#f97316',
          contrast: '#0f172a',
        },
        background: {
          DEFAULT: '#f1f5f9',
          muted: '#e2e8f0',
          dark: '#0f172a',
        },
        surface: {
          DEFAULT: '#ffffff',
          dark: '#1e293b',
        },
      },
      boxShadow: {
        'soft-lg': '0 20px 45px -20px rgba(37, 99, 235, 0.35)',
      },
      fontFamily: {
        sans: ['"Noto Sans SC"', '"Inter"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
