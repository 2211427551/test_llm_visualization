'use client';

import { useState } from 'react';
import { useVisualizationStore } from '@/store/visualizationStore';
import { Card, Button } from './ui';
import { Edit3, Sparkles, Play, ChevronDown } from 'lucide-react';

export default function InputModule() {
  const { inputText, setInputText, initializeComputation, isLoading, isInitialized, config, setConfig } = useVisualizationStore();
  
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await initializeComputation();
  };

  return (
    <Card className="mb-6">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 bg-purple-500/20 rounded-lg flex items-center justify-center">
          <Edit3 className="w-5 h-5 text-purple-400" />
        </div>
        <h3 className="text-lg font-semibold text-white">输入文本</h3>
      </div>
      
      <form onSubmit={handleSubmit}>
        <div className="relative mb-4">
          <textarea
            className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-4 py-3 text-white placeholder-slate-500 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none resize-none transition-all"
            rows={4}
            placeholder="输入您想要可视化的文本，例如：hello world"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isInitialized}
          />
          <div className="absolute bottom-3 right-3 text-xs text-slate-500">
            {inputText.length} 字符
          </div>
        </div>

        {/* Advanced Configuration Toggle */}
        <div className="mb-4">
          <button
            type="button"
            className="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300 transition-colors"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <ChevronDown 
              className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
            />
            {showAdvanced ? '隐藏' : '显示'}高级配置
          </button>
        </div>

        {/* Advanced Configuration */}
        {showAdvanced && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4 p-4 bg-slate-900/30 rounded-lg border border-slate-700/50">
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                词汇表大小
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                value={config.n_vocab}
                onChange={(e) => setConfig({ n_vocab: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                嵌入维度
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                value={config.n_embd}
                onChange={(e) => setConfig({ n_embd: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                层数
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                value={config.n_layer}
                onChange={(e) => setConfig({ n_layer: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                注意力头数
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                value={config.n_head}
                onChange={(e) => setConfig({ n_head: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                注意力头维度
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                value={config.d_k}
                onChange={(e) => setConfig({ d_k: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">
                最大序列长度
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500/20 focus:border-purple-500"
                value={config.max_seq_len}
                onChange={(e) => setConfig({ max_seq_len: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <Button 
            type="button"
            variant="outline" 
            className="w-full"
            disabled={isInitialized}
          >
            <Sparkles className="w-4 h-4" />
            使用示例
          </Button>
          <Button
            type="submit"
            variant="primary"
            className="w-full"
            disabled={isLoading || isInitialized || !inputText.trim()}
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                初始化中...
              </>
            ) : isInitialized ? (
              '已初始化'
            ) : (
              <>
                <Play className="w-4 h-4" />
                开始分析
              </>
            )}
          </Button>
        </div>
      </form>
    </Card>
  );
}
