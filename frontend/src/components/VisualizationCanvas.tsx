'use client';

import { useVisualizationStore } from '@/store/visualizationStore';

export default function VisualizationCanvas() {
  const { isInitialized, currentStepData, tokenTexts } = useVisualizationStore();

  if (!isInitialized) {
    return (
      <div className="bg-gray-100 rounded-lg shadow-md p-8 flex items-center justify-center min-h-[400px]">
        <div className="text-center text-gray-500">
          <svg
            className="w-16 h-16 mx-auto mb-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <p className="text-lg font-medium">可视化画布</p>
          <p className="text-sm">等待数据初始化...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6 min-h-[400px]">
      <h3 className="text-xl font-bold text-gray-800 mb-4">可视化区域</h3>
      
      {/* Tokens Display */}
      <div className="mb-6">
        <h4 className="text-sm font-semibold text-gray-700 mb-2">Token 序列:</h4>
        <div className="flex flex-wrap gap-2">
          {tokenTexts.map((token, idx) => (
            <span
              key={idx}
              className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium"
            >
              {token}
            </span>
          ))}
        </div>
      </div>

      {/* Current Step Data */}
      {currentStepData && (
        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">当前步骤数据:</h4>
            <div className="space-y-2 text-sm">
              <div className="flex">
                <span className="font-medium text-gray-600 w-32">步骤类型:</span>
                <span className="text-gray-800">{currentStepData.step_type}</span>
              </div>
              <div className="flex">
                <span className="font-medium text-gray-600 w-32">层索引:</span>
                <span className="text-gray-800">{currentStepData.layer_index}</span>
              </div>
              <div className="flex">
                <span className="font-medium text-gray-600 w-32">描述:</span>
                <span className="text-gray-800">{currentStepData.description}</span>
              </div>
            </div>
          </div>

          {/* Data Preview */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="text-sm font-semibold text-gray-700 mb-2">数据预览 (JSON):</h4>
            <div className="bg-gray-900 text-green-400 rounded p-3 overflow-auto max-h-96 text-xs font-mono">
              <pre>{JSON.stringify(currentStepData, null, 2)}</pre>
            </div>
          </div>

          {/* Metadata */}
          {currentStepData.metadata && Object.keys(currentStepData.metadata).length > 0 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-semibold text-gray-700 mb-2">元数据:</h4>
              <div className="bg-gray-900 text-green-400 rounded p-3 overflow-auto max-h-48 text-xs font-mono">
                <pre>{JSON.stringify(currentStepData.metadata, null, 2)}</pre>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Placeholder for D3.js */}
      <div className="mt-6 border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
        <p className="text-gray-500 text-sm">
          📊 D3.js 可视化区域预留
        </p>
        <p className="text-gray-400 text-xs mt-2">
          在后续阶段将集成交互式可视化
        </p>
      </div>
    </div>
  );
}
