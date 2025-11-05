import { useState } from 'react'

interface ModelParameters {
  learningRate: number
  batchSize: number
  epochs: number
  dropout: number
}

const LeftPanel = () => {
  const [inputText, setInputText] = useState('')
  const [showParameters, setShowParameters] = useState(true)
  const [currentStep, setCurrentStep] = useState(1)
  const [parameters, setParameters] = useState<ModelParameters>({
    learningRate: 0.001,
    batchSize: 32,
    epochs: 10,
    dropout: 0.1,
  })

  const handleParameterChange = (key: keyof ModelParameters, value: number) => {
    setParameters((prev) => ({ ...prev, [key]: value }))
  }

  const handleSubmit = () => {
    console.log('Submitting with:', { inputText, parameters })
  }

  const handlePreviousStep = () => {
    setCurrentStep((prev) => Math.max(1, prev - 1))
  }

  const handleNextStep = () => {
    setCurrentStep((prev) => prev + 1)
  }

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">输入与控制</h2>
      </div>

      {/* Text Input */}
      <div className="flex-1">
        <label className="mb-2 block text-sm font-medium text-slate-700 dark:text-slate-300">
          文本输入
        </label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="请输入要处理的文本..."
          className="h-32 w-full rounded-md border border-slate-300 px-3 py-2 text-sm placeholder-slate-400 focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary dark:border-slate-600 dark:bg-slate-700 dark:text-slate-100 dark:placeholder-slate-400"
        />
      </div>

      {/* Model Parameters */}
      <div className="flex-1">
        <div className="mb-3 flex items-center justify-between">
          <label className="text-sm font-medium text-slate-700 dark:text-slate-300">模型参数</label>
          <button
            onClick={() => setShowParameters(!showParameters)}
            className="hover:text-primary-dark text-xs text-primary dark:text-primary-contrast dark:hover:text-primary-contrast"
          >
            {showParameters ? '隐藏' : '显示'}
          </button>
        </div>

        {showParameters && (
          <div className="space-y-3">
            <div>
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400">
                学习率: {parameters.learningRate}
              </label>
              <input
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
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400">
                批次大小: {parameters.batchSize}
              </label>
              <input
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
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400">
                训练轮数: {parameters.epochs}
              </label>
              <input
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
              <label className="mb-1 block text-xs text-slate-600 dark:text-slate-400">
                Dropout: {parameters.dropout}
              </label>
              <input
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

      {/* Control Buttons */}
      <div className="space-y-3">
        <button
          onClick={handleSubmit}
          className="hover:bg-primary-dark w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-contrast focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2 dark:bg-primary-contrast dark:text-primary dark:hover:bg-primary"
        >
          提交
        </button>

        <div className="flex gap-2">
          <button
            onClick={handlePreviousStep}
            disabled={currentStep <= 1}
            className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            上一步
          </button>
          <span className="flex items-center px-3 text-sm text-slate-600 dark:text-slate-400">
            步骤 {currentStep}
          </span>
          <button
            onClick={handleNextStep}
            className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
          >
            下一步
          </button>
        </div>
      </div>
    </div>
  )
}

export default LeftPanel
