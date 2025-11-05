import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

void i18n.use(initReactI18next).init({
  fallbackLng: 'zh-CN',
  lng: 'zh-CN',
  resources: {
    'zh-CN': {
      translation: {
        layout: {
          title: '数据运营面板',
          subtitle: '掌握实时指标，洞察业务趋势。',
          footer: '© 2025 数据洞察平台',
          themeToggle: '切换主题',
        },
        actions: {
          switchToDark: '切换至深色模式',
          switchToLight: '切换至浅色模式',
        },
        dashboard: {
          welcome: '欢迎回来！',
          highlight: '关键指标概览',
          chartTitle: '七日访问量趋势',
          callout: '通过数据可视化洞察，帮助你快速做出业务判断。',
          promo: '团队可在此快速追踪核心指标，确保沟通一致。',
          trendHint: '较上周',
          tips: ['监控关键路径，及时发现波动。', '结合定性反馈完善产品体验。'],
          metrics: {
            users: '新增用户',
            retention: '次日留存率',
            conversion: '转化率',
          },
        },
      },
    },
  },
  interpolation: {
    escapeValue: false,
  },
})

export default i18n
