/**
 * 左侧面板组件
 *
 * 功能：
 * - 模型配置和初始化
 * - 文本输入和推理控制
 * - 状态管理和错误处理
 * - 与可视化状态的集成
 */

import { useCallback, useEffect, useMemo, useState } from 'react'
import { forwardInference, initializeModel } from '../services/api'
import type { ModelConfig } from '../types/api'
import { useVisualizationState } from '../hooks/useVisualizationState'

const formatTimestamp = (value: string | undefined) => {
  if (!value) {
    return '未知时间'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString('zh-CN', { hour12: false })
}

const LeftPanel = () => {
  // 状态管理
  const [inputText, setInputText] = useState('') // 输入文本
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null) // 模型配置
  const [initLoading, setInitLoading] = useState(false) // 初始化加载状态
  const [forwardLoading, setForwardLoading] = useState(false) // 推理加载状态
  const [initError, setInitError] = useState<string | null>(null) // 初始化错误信息
  const [forwardError, setForwardError] = useState<string | null>(null) // 推理错误信息
  const [statusMessage, setStatusMessage] = useState<string | null>(null) // 状态消息

  const {
    steps,
    currentStep,
    currentStepIndex,
    stepCount,
    canGoToNext,
    canGoToPrevious,
    goToNextStep,
    goToPreviousStep,
    setStepByIndex,
    runtimeSummary,
    updateVisualization,
  } = useVisualizationState()

  const progressPercentage = useMemo(() => {
    if (stepCount === 0) {
      return 0
    }
    return Math.round(((currentStepIndex + 1) / stepCount) * 100)
  }, [currentStepIndex, stepCount])

  const loadModelConfig = useCallback(async () => {
    setInitLoading(true)
    setInitError(null)
    try {
      const response = await initializeModel()
      setModelConfig(response.config)
      setStatusMessage('模型初始化成功。')
    } catch (error) {
      const message = error instanceof Error ? error.message : '模型初始化失败，请稍后重试。'
      setInitError(message)
    } finally {
      setInitLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadModelConfig()
  }, [loadModelConfig])

  const handleSubmit = useCallback(async () => {
    if (forwardLoading) {
      return
    }
    if (!inputText.trim()) {
      setForwardError('请输入要处理的文本。')
      return
    }
    if (!modelConfig) {
      setForwardError('模型尚未初始化完成，请稍后重试。')
      return
    }

    setForwardLoading(true)
    setForwardError(null)
    setStatusMessage(null)
    try {
      const response = await forwardInference({ text: inputText, captureData: true })
      if (response.capturedData) {
        updateVisualization({
          steps: response.capturedData.steps,
          runtime: response.capturedData.runtime,
          tokenSequence: response.capturedData.tokenSequence,
        })
      }
      setStatusMessage('前向推理完成，已同步最新可视化数据。')
    } catch (error) {
      const message = error instanceof Error ? error.message : '推理失败，请稍后重试。'
      setForwardError(message)
    } finally {
      setForwardLoading(false)
    }
  }, [forwardLoading, inputText, modelConfig, updateVisualization])

  const modelConfigItems = useMemo(() => {
    if (!modelConfig) {
      return []
    }
    return [
      { label: '模型名称', value: modelConfig.modelName },
      { label: '词表大小', value: modelConfig.vocabSize },
      { label: '上下文长度', value: modelConfig.contextSize },
      { label: '层数 / 注意力头', value: `${modelConfig.nLayer} 层 · ${modelConfig.nHead} 头` },
      { label: '嵌入维度', value: modelConfig.nEmbed },
      { label: 'Dropout', value: modelConfig.dropout },
      {
        label: '稀疏注意力',
        value: modelConfig.useSparseAttention ? '已启用' : '未启用',
      },
      {
        label: 'MoE 设置',
        value: modelConfig.useMoe
          ? `${modelConfig.moeNumExperts} 专家 · TopK ${modelConfig.moeTopK}`
          : '未启用',
      },
      { label: '最近初始化', value: formatTimestamp(modelConfig.initializedAt) },
    ]
  }, [modelConfig])

  return (
    <div className="flex h-full flex-col gap-4" aria-label="左侧控制面板">
      <div>
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">输入与控制</h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          配置推理参数并逐步查看模型层级。
        </p>
      </div>

      <div className="space-y-3" aria-label="模型状态">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">模型配置</span>
          <div className="flex items-center gap-2">
            {initLoading && <span className="text-xs text-primary">加载中...</span>}
            <button
              type="button"
              onClick={() => void loadModelConfig()}
              className="rounded-md border border-slate-300 px-3 py-1 text-xs font-medium text-slate-600 hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
              disabled={initLoading}
            >
              刷新模型
            </button>
          </div>
        </div>

        {initError ? (
          <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-500/40 dark:bg-red-900/20 dark:text-red-300">
            {initError}
          </div>
        ) : (
          <dl className="grid grid-cols-2 gap-2 rounded-lg border border-slate-200 p-3 text-xs text-slate-600 dark:border-slate-700 dark:text-slate-300">
            {modelConfigItems.length === 0 ? (
              <div className="col-span-2 text-center text-slate-400">等待初始化...</div>
            ) : (
              modelConfigItems.map((item) => (
                <div key={item.label} className="flex flex-col gap-1">
                  <dt className="font-medium text-slate-500 dark:text-slate-400">{item.label}</dt>
                  <dd className="text-slate-900 dark:text-slate-100">{item.value}</dd>
                </div>
              ))
            )}
          </dl>
        )}

        {statusMessage && !initError && (
          <div className="rounded-md bg-emerald-50 px-3 py-2 text-xs text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-300">
            {statusMessage}
          </div>
        )}
      </div>

      <div className="space-y-2" aria-label="文本输入">
        <label
          className="block text-sm font-medium text-slate-700 dark:text-slate-300"
          htmlFor="model-input"
        >
          文本输入
        </label>
        <textarea
          id="model-input"
          value={inputText}
          onChange={(event) => setInputText(event.target.value)}
          placeholder="请输入要处理的文本..."
          className="h-32 w-full rounded-md border border-slate-300 px-3 py-2 text-sm placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary dark:border-slate-600 dark:bg-slate-700 dark:text-slate-100 dark:placeholder-slate-400"
          aria-describedby="input-helper-text"
        />
        <p id="input-helper-text" className="text-xs text-slate-500 dark:text-slate-400">
          提交后会触发前向推理，并实时更新中栏与右栏的可视化信息。
        </p>
        {forwardError && (
          <p className="text-xs text-red-500" role="alert">
            {forwardError}
          </p>
        )}
        <button
          type="button"
          onClick={() => void handleSubmit()}
          className="hover:bg-primary-dark w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-contrast focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-primary-contrast dark:text-primary dark:hover:bg-primary"
          disabled={forwardLoading || initLoading}
        >
          {forwardLoading ? '推理中...' : '提交推理'}
        </button>
      </div>

      <div
        className="rounded-lg border border-slate-200 p-3 dark:border-slate-700"
        aria-labelledby="step-status-heading"
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <p
              id="step-status-heading"
              className="text-xs font-medium text-slate-600 dark:text-slate-400"
            >
              当前步骤
            </p>
            <p
              role="status"
              aria-live="polite"
              className="mt-1 text-sm font-semibold text-slate-900 dark:text-slate-50"
            >
              第 {currentStepIndex + 1} 步 · {currentStep.name}
            </p>
          </div>
          <div className="text-right text-xs text-slate-500 dark:text-slate-400">
            <span>进度 {progressPercentage}%</span>
            {runtimeSummary && (
              <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">
                最近推理：{formatTimestamp(runtimeSummary.capturedAt)}
              </div>
            )}
          </div>
        </div>

        <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
          {currentStep.description}
        </p>

        <div
          className="mt-2 h-2 w-full rounded-full bg-slate-200 dark:bg-slate-700"
          aria-hidden="true"
        >
          <div
            className="h-2 rounded-full bg-gradient-to-r from-primary to-secondary transition-all duration-500"
            style={{ width: `${progressPercentage}%` }}
          />
        </div>

        <div className="mt-3 flex items-center gap-2" role="group" aria-label="逐步执行控制">
          <button
            type="button"
            onClick={goToPreviousStep}
            disabled={!canGoToPrevious}
            className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            上一步
          </button>
          <span className="flex items-center px-2 text-xs text-slate-500 dark:text-slate-400">
            步骤 {currentStepIndex + 1}/{stepCount}
          </span>
          <button
            type="button"
            onClick={goToNextStep}
            disabled={!canGoToNext}
            className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            下一步
          </button>
        </div>

        <ol className="mt-3 max-h-48 space-y-1 overflow-y-auto pr-1" aria-label="步骤跳转">
          {steps.map((step, index) => {
            const isActive = index === currentStepIndex
            return (
              <li key={step.id}>
                <button
                  type="button"
                  onClick={() => setStepByIndex(index)}
                  aria-pressed={isActive}
                  className={`w-full rounded-md border px-3 py-2 text-left text-xs transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 dark:border-slate-700 ${
                    isActive
                      ? 'border-primary bg-primary/10 text-primary dark:text-primary-contrast'
                      : 'border-slate-200 text-slate-600 hover:border-primary/40 hover:text-primary dark:text-slate-300'
                  }`}
                >
                  第 {index + 1} 步 · {step.name}
                </button>
              </li>
            )
          })}
        </ol>
      </div>
    </div>
  )
}

export default LeftPanel
