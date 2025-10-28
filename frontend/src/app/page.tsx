'use client';

import InputModule from '@/components/InputModule';
import ControlPanel from '@/components/ControlPanel';
import VisualizationCanvas from '@/components/VisualizationCanvas';
import ExplanationPanel from '@/components/ExplanationPanel';
import { useVisualizationStore } from '@/store/visualizationStore';

export default function Home() {
  const { error, reset } = useVisualizationStore();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Transformer 计算可视化
              </h1>
              <p className="mt-1 text-sm text-gray-600">
                逐步可视化 Transformer 模型的计算过程
              </p>
            </div>
            <button
              onClick={reset}
              className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg transition duration-200"
            >
              重置
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg flex items-start">
            <svg
              className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <div>
              <p className="font-semibold">错误</p>
              <p className="text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Input Module */}
        <InputModule />

        {/* Control Panel */}
        <ControlPanel />

        {/* Visualization and Explanation */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Visualization Canvas - 70% width on large screens */}
          <div className="lg:col-span-2">
            <VisualizationCanvas />
          </div>

          {/* Explanation Panel - 30% width on large screens */}
          <div className="lg:col-span-1">
            <ExplanationPanel />
          </div>
        </div>
      </main>

      {/* Demo Links */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">独立演示</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <a
              href="/demo"
              className="block p-4 bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200 transition-colors"
            >
              <h3 className="font-semibold text-blue-900 mb-2">Token 嵌入演示</h3>
              <p className="text-sm text-blue-700">词元化和嵌入过程可视化</p>
            </a>
            <a
              href="/attention-demo"
              className="block p-4 bg-green-50 hover:bg-green-100 rounded-lg border border-green-200 transition-colors"
            >
              <h3 className="font-semibold text-green-900 mb-2">多头注意力演示</h3>
              <p className="text-sm text-green-700">Multi-Head Attention 完整流程</p>
            </a>
            <a
              href="/moe-demo"
              className="block p-4 bg-purple-50 hover:bg-purple-100 rounded-lg border border-purple-200 transition-colors"
            >
              <h3 className="font-semibold text-purple-900 mb-2">MoE FFN 演示 🆕</h3>
              <p className="text-sm text-purple-700">混合专家模型前馈网络</p>
            </a>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-600">
            Transformer 计算可视化工具 - 教育演示版本
          </p>
        </div>
      </footer>
    </div>
  );
}
