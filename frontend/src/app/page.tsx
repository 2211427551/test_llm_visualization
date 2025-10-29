'use client';

import InputModule from '@/components/InputModule';
import ControlPanel from '@/components/ControlPanel';
import VisualizationCanvas from '@/components/VisualizationCanvas';
import ExplanationPanel from '@/components/ExplanationPanel';
import { Header } from '@/components/Header';
import { HelpButton } from '@/components/HelpDialog';
import { useVisualizationStore } from '@/store/visualizationStore';
import { AlertCircle } from 'lucide-react';

export default function Home() {
  const { error, reset } = useVisualizationStore();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <Header onReset={reset} />

      {/* Main Content */}
      <main className="container mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Error Display */}
        {error && (
          <div className="mb-6 bg-red-500/10 border border-red-500/50 text-red-200 px-4 py-3 rounded-lg flex items-start animate-slide-in-up">
            <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0 mt-0.5" />
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

        {/* Visualization and Explanation Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Visualization Canvas - 8 columns */}
          <div className="lg:col-span-8">
            <VisualizationCanvas />
          </div>

          {/* Explanation Panel - 4 columns */}
          <div className="lg:col-span-4">
            <ExplanationPanel />
          </div>
        </div>
      </main>

      {/* Demo Links */}
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 mt-8">
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 shadow-xl">
          <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
              独立演示
            </span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <a
              href="/demo"
              className="block p-4 bg-blue-500/10 hover:bg-blue-500/20 rounded-lg border border-blue-500/30 hover:border-blue-500/50 transition-all duration-200 group"
            >
              <h3 className="font-semibold text-blue-300 mb-2 group-hover:text-blue-200">
                Token 嵌入演示
              </h3>
              <p className="text-sm text-slate-400">词元化和嵌入过程可视化</p>
            </a>
            <a
              href="/attention-demo"
              className="block p-4 bg-green-500/10 hover:bg-green-500/20 rounded-lg border border-green-500/30 hover:border-green-500/50 transition-all duration-200 group"
            >
              <h3 className="font-semibold text-green-300 mb-2 group-hover:text-green-200">
                多头注意力演示
              </h3>
              <p className="text-sm text-slate-400">Multi-Head Attention 完整流程</p>
            </a>
            <a
              href="/sparse-attention-demo"
              className="block p-4 bg-orange-500/10 hover:bg-orange-500/20 rounded-lg border border-orange-500/30 hover:border-orange-500/50 transition-all duration-200 group"
            >
              <h3 className="font-semibold text-orange-300 mb-2 group-hover:text-orange-200">
                稀疏注意力演示
              </h3>
              <p className="text-sm text-slate-400">Sparse Attention 效率优化</p>
            </a>
            <a
              href="/moe-demo"
              className="block p-4 bg-purple-500/10 hover:bg-purple-500/20 rounded-lg border border-purple-500/30 hover:border-purple-500/50 transition-all duration-200 group"
            >
              <h3 className="font-semibold text-purple-300 mb-2 group-hover:text-purple-200">
                MoE FFN 演示
              </h3>
              <p className="text-sm text-slate-400">混合专家模型前馈网络</p>
            </a>
            <a
              href="/output-layer-demo"
              className="block p-4 bg-amber-500/10 hover:bg-amber-500/20 rounded-lg border border-amber-500/30 hover:border-amber-500/50 transition-all duration-200 group"
            >
              <h3 className="font-semibold text-amber-300 mb-2 group-hover:text-amber-200 flex items-center gap-2">
                输出层演示
                <span className="text-xs bg-amber-500/20 px-2 py-0.5 rounded-full">NEW</span>
              </h3>
              <p className="text-sm text-slate-400">Logits、Softmax、预测结果</p>
            </a>
            <a
              href="/examples"
              className="block p-4 bg-pink-500/10 hover:bg-pink-500/20 rounded-lg border border-pink-500/30 hover:border-pink-500/50 transition-all duration-200 group"
            >
              <h3 className="font-semibold text-pink-300 mb-2 group-hover:text-pink-200">
                示例集合
              </h3>
              <p className="text-sm text-slate-400">所有可视化组件示例</p>
            </a>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-slate-900/50 backdrop-blur-sm border-t border-slate-700/50 mt-12">
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-slate-400">
            Transformer 计算可视化工具 - 教育演示版本
          </p>
        </div>
      </footer>

      {/* Help Button */}
      <HelpButton />
    </div>
  );
}
