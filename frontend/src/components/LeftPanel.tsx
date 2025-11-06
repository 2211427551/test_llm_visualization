import { useMemo, useState } from 'react'
import { useVisualizationState } from '../hooks/useVisualizationState'

interface ModelParameters {
  learningRate: number
  batchSize: number
  epochs: number
  dropout: number
}

const LeftPanel = () => {
  const [inputText, setInputText] = useState('')
  const [showParameters, setShowParameters] = useState(true)
  const [parameters, setParameters] = useState<ModelParameters>({
    learningRate: 0.001,
    batchSize: 32,
    epochs: 10,
    dropout: 0.1,
  })

  const {
    steps,
    currentStep,
    currentStepIndex,
    stepCount,
    canGoToPrevious,
    canGoToNext,
    goToPreviousStep,
    goToNextStep,
    setStepByIndex,
  } = useVisualizationState()

  const progressPercentage = useMemo(() => {
    if (stepCount === 0) {
      return 0
    }
    return Math.round(((currentStepIndex + 1) / stepCount) * 100)
  }, [currentStepIndex, stepCount])

  const handleParameterChange = (key: keyof ModelParameters, value: number) => {
    setParameters((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = () => {
    console.log('Submitting with:', { inputText, parameters, step: currentStep.id })
  }

  return (
    <div className="flex h-full flex-col gap-4" aria-label="左侧控制面板">
      <div>
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">输入与控制</h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          配置推理参数并逐步查看模型层级。
        </p>
      </div>

      <div className="flex-1" aria-label="文本输入">
        <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-300" htmlFor="model-input">
          文本输入
        </label>
        <textarea
          id="model-input"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="请输入要处理的文本..."
          className="h-32 w-full rounded-md border border-slate-300 px-3 py-2 text-sm placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary dark:border-slate-600 dark:bg-slate-700 dark:text-slate-100 dark:placeholder-slate-400"
          aria-describedby="input-helper-text"
        />
        <p id="input-helper-text" className="mt-2 text-xs text-slate-500 dark:text-slate-400">
          支持逐步执行，可在下方切换不同推理阶段观察变化。
        </p>
      </div>

      <div className="flex-1" aria-label="模型参数">
        <div className="mb-3 flex items-center justify-between">
          <label className="text-sm font-medium text-slate-700 dark:text-slate-300">模型参数</label>
          <button
            type="button"
            onClick={() => setShowParameters(!showParameters)}
            className="text-xs text-primary hover:text-primary-dark focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 dark:text-primary-contrast"
            aria-expanded={showParameters}
            aria-controls="parameter-controls"
          >
            {showParameters ? '隐藏' : '显示'}
          </button>
        </div>

        {showParameters && (
          <div id="parameter-controls" className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400" htmlFor="param-learning-rate">
                学习率: {parameters.learningRate}
              </label>
              <input
                id="param-learning-rate"
                type="range"
                min="0.0001"
                max="0.01"
                step="0.0001"
                value={parameters.learningRate}
                onChange={(e) => handleParameterChange('learningRate', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400" htmlFor="param-batch-size">
                批次大小: {parameters.batchSize}
              </label>
              <input
                id="param-batch-size"
                type="range"
                min="8"
                max="128"
                step="8"
                value={parameters.batchSize}
                onChange={(e) => handleParameterChange('batchSize', parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400" htmlFor="param-epochs">
                训练轮数: {parameters.epochs}
              </label>
              <input
                id="param-epochs"
                type="range"
                min="1"
                max="50"
                step="1"
                value={parameters.epochs}
                onChange={(e) => handleParameterChange('epochs', parseInt(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400" htmlFor="param-dropout">
                Dropout: {parameters.dropout}
              </label>
              <input
                id="param-dropout"
                type="range"
                min="0"
                max="0.5"
                step="0.05"
                value={parameters.dropout}
                onChange={(e) => handleParameterChange('dropout', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        )}
      </div>

      <div className="space-y-3" aria-label="执行控制">
        <button
          type="button"
          onClick={handleSubmit}
          className="hover:bg-primary-dark w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-contrast focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 dark:bg-primary-contrast dark:text-primary dark:hover:bg-primary"
        >
          提交
        </button>

        <div className="rounded-lg border border-slate-200 p-3 dark:border-slate-700" aria-labelledby="step-status-heading">
          <div className="flex items-start justify-between gap-2">
            <div>
              <p id="step-status-heading" className="text-xs font-medium text-slate-600 dark:text-slate-400">
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
            </div>
          </div>

          <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-400">
            {currentStep.description}
          </p>

          <div className="mt-2 h-2 w-full rounded-full bg-slate-200 dark:bg-slate-700" aria-hidden="true">
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

          <ol className="mt-3 space-y-1" aria-label="步骤跳转">
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
    </div>
  )
}

export default LeftPanel
