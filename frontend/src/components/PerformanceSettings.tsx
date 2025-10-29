'use client';

import React, { useState } from 'react';
import { usePerformanceMode, usePerformanceMonitor } from '@/hooks/usePerformanceMode';

export const PerformanceSettings: React.FC = () => {
  const { mode, setMode, settings } = usePerformanceMode();
  const { fps, memoryUsage } = usePerformanceMonitor();
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-200 rounded-lg transition-colors"
        aria-label="Performance settings"
        aria-expanded={isOpen}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 10V3L4 14h7v7l9-11h-7z"
          />
        </svg>
        <span className="text-sm font-medium">性能</span>
        <span className={`text-xs px-2 py-0.5 rounded ${
          mode === 'high' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
          mode === 'balanced' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
          'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
        }`}>
          {mode === 'high' ? '高' : mode === 'balanced' ? '平衡' : '低'}
        </span>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
            aria-hidden="true"
          />

          {/* Dropdown Menu */}
          <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4 z-20">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
              性能设置
            </h3>

            {/* Performance Modes */}
            <div className="space-y-2 mb-4">
              <button
                onClick={() => setMode('high')}
                className={`w-full px-3 py-2 text-left rounded-lg transition-colors ${
                  mode === 'high'
                    ? 'bg-green-100 dark:bg-green-900 border-2 border-green-500'
                    : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600'
                }`}
              >
                <div className="font-medium text-gray-900 dark:text-gray-100">高性能模式</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  完整动画效果，适合高性能设备
                </div>
              </button>

              <button
                onClick={() => setMode('balanced')}
                className={`w-full px-3 py-2 text-left rounded-lg transition-colors ${
                  mode === 'balanced'
                    ? 'bg-blue-100 dark:bg-blue-900 border-2 border-blue-500'
                    : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600'
                }`}
              >
                <div className="font-medium text-gray-900 dark:text-gray-100">平衡模式</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  优化的动画效果，推荐使用
                </div>
              </button>

              <button
                onClick={() => setMode('low')}
                className={`w-full px-3 py-2 text-left rounded-lg transition-colors ${
                  mode === 'low'
                    ? 'bg-orange-100 dark:bg-orange-900 border-2 border-orange-500'
                    : 'bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600'
                }`}
              >
                <div className="font-medium text-gray-900 dark:text-gray-100">低性能模式</div>
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  简化动画，适合低性能设备
                </div>
              </button>
            </div>

            {/* Current Settings */}
            <div className="border-t border-gray-200 dark:border-gray-700 pt-3 mb-3">
              <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                <div className="flex justify-between">
                  <span>动画时长:</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {settings.animationDuration}ms
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>最大元素:</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {settings.maxElements.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>过渡效果:</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {settings.enableTransitions ? '启用' : '禁用'}
                  </span>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
              <div className="text-xs font-semibold text-gray-900 dark:text-gray-100 mb-2">
                实时性能
              </div>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-gray-600 dark:text-gray-400">帧率 (FPS)</span>
                    <span className={`text-xs font-medium ${
                      fps >= 50 ? 'text-green-600 dark:text-green-400' :
                      fps >= 30 ? 'text-yellow-600 dark:text-yellow-400' :
                      'text-red-600 dark:text-red-400'
                    }`}>
                      {fps}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full transition-all ${
                        fps >= 50 ? 'bg-green-500' :
                        fps >= 30 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${Math.min((fps / 60) * 100, 100)}%` }}
                    />
                  </div>
                </div>

                {memoryUsage > 0 && (
                  <div>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-xs text-gray-600 dark:text-gray-400">内存使用</span>
                      <span className={`text-xs font-medium ${
                        memoryUsage < 70 ? 'text-green-600 dark:text-green-400' :
                        memoryUsage < 85 ? 'text-yellow-600 dark:text-yellow-400' :
                        'text-red-600 dark:text-red-400'
                      }`}>
                        {memoryUsage}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                      <div
                        className={`h-1.5 rounded-full transition-all ${
                          memoryUsage < 70 ? 'bg-green-500' :
                          memoryUsage < 85 ? 'bg-yellow-500' :
                          'bg-red-500'
                        }`}
                        style={{ width: `${memoryUsage}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
