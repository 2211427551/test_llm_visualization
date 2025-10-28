'use client';

import { useState } from 'react';
import { useVisualizationStore } from '@/store/visualizationStore';

export default function InputModule() {
  const { inputText, setInputText, initializeComputation, isLoading, isInitialized, config, setConfig } = useVisualizationStore();
  
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await initializeComputation();
  };

  return (
    <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">输入文本</h2>
      
      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <textarea
            className="w-full px-4 py-3 text-gray-700 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={4}
            placeholder="请输入要分析的文本..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isInitialized}
          />
        </div>

        {/* Advanced Configuration Toggle */}
        <div className="mb-4">
          <button
            type="button"
            className="text-sm text-blue-600 hover:text-blue-800 underline"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '隐藏' : '显示'}高级配置
          </button>
        </div>

        {/* Advanced Configuration */}
        {showAdvanced && (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-4 p-4 bg-gray-50 rounded-lg">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                词汇表大小
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={config.n_vocab}
                onChange={(e) => setConfig({ n_vocab: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                嵌入维度
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={config.n_embd}
                onChange={(e) => setConfig({ n_embd: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                层数
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={config.n_layer}
                onChange={(e) => setConfig({ n_layer: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                注意力头数
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={config.n_head}
                onChange={(e) => setConfig({ n_head: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                注意力头维度
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={config.d_k}
                onChange={(e) => setConfig({ d_k: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                最大序列长度
              </label>
              <input
                type="number"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                value={config.max_seq_len}
                onChange={(e) => setConfig({ max_seq_len: parseInt(e.target.value) })}
                disabled={isInitialized}
              />
            </div>
          </div>
        )}

        <button
          type="submit"
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-200 ease-in-out disabled:bg-gray-400 disabled:cursor-not-allowed"
          disabled={isLoading || isInitialized || !inputText.trim()}
        >
          {isLoading ? '初始化中...' : isInitialized ? '已初始化' : '开始计算'}
        </button>
      </form>
    </div>
  );
}
