'use client';

import React, { useState, useEffect } from 'react';

interface HelpDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HelpDialog: React.FC<HelpDialogProps> = ({ isOpen, onClose }) => {
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="help-dialog-title"
    >
      <div
        className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <h2 id="help-dialog-title" className="text-2xl font-bold text-gray-900">
            使用帮助
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Close help dialog"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="px-6 py-4 space-y-6">
          {/* Quick Start */}
          <section>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">🚀 快速开始</h3>
            <ol className="list-decimal list-inside space-y-2 text-gray-700">
              <li>在输入框中输入文本</li>
              <li>点击"初始化"按钮开始分析</li>
              <li>使用控制面板播放/暂停动画</li>
              <li>查看解释面板了解每一步的详细信息</li>
            </ol>
          </section>

          {/* Keyboard Shortcuts */}
          <section>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">⌨️ 键盘快捷键</h3>
            <div className="bg-gray-50 rounded-lg p-4 space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-700">播放/暂停</span>
                <kbd className="px-2 py-1 bg-white border border-gray-300 rounded text-sm font-mono">
                  Space
                </kbd>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-700">下一步</span>
                <kbd className="px-2 py-1 bg-white border border-gray-300 rounded text-sm font-mono">
                  →
                </kbd>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-700">上一步</span>
                <kbd className="px-2 py-1 bg-white border border-gray-300 rounded text-sm font-mono">
                  ←
                </kbd>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-700">重置</span>
                <kbd className="px-2 py-1 bg-white border border-gray-300 rounded text-sm font-mono">
                  R
                </kbd>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-700">关闭对话框</span>
                <kbd className="px-2 py-1 bg-white border border-gray-300 rounded text-sm font-mono">
                  Esc
                </kbd>
              </div>
            </div>
          </section>

          {/* Features */}
          <section>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">✨ 功能说明</h3>
            <div className="space-y-3">
              <div>
                <h4 className="font-medium text-gray-900">可视化模式</h4>
                <p className="text-sm text-gray-600">
                  支持D3.js动画模式和数据视图模式，可随时切换
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">导出功能</h4>
                <p className="text-sm text-gray-600">
                  可以导出当前可视化为SVG或PNG格式，也可导出计算数据为JSON
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">性能模式</h4>
                <p className="text-sm text-gray-600">
                  根据设备自动调整性能，也可手动切换高性能/平衡/低性能模式
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">暗黑模式</h4>
                <p className="text-sm text-gray-600">
                  支持明亮/暗黑主题切换，保护您的眼睛
                </p>
              </div>
            </div>
          </section>

          {/* Tips */}
          <section>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">💡 使用技巧</h3>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li>悬停在矩阵单元格上可以查看具体数值</li>
              <li>使用较短的文本可以获得更快的计算速度</li>
              <li>在低性能模式下，动画会更快速但更简洁</li>
              <li>查看示例集合页面了解所有可视化组件</li>
            </ul>
          </section>

          {/* Links */}
          <section>
            <h3 className="text-lg font-semibold text-gray-900 mb-3">🔗 相关资源</h3>
            <div className="space-y-2">
              <a
                href="https://arxiv.org/abs/1706.03762"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-blue-600 hover:text-blue-800 hover:underline"
              >
                📄 Attention Is All You Need (原始论文)
              </a>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="block text-blue-600 hover:text-blue-800 hover:underline"
              >
                💻 GitHub 仓库
              </a>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-gray-50 border-t border-gray-200 px-6 py-4">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors font-medium"
          >
            开始使用
          </button>
        </div>
      </div>
    </div>
  );
};

export const HelpButton: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-40 w-14 h-14 bg-blue-500 hover:bg-blue-600 text-white rounded-full shadow-lg flex items-center justify-center transition-all hover:scale-110"
        aria-label="Open help"
        title="帮助 (快捷键)"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      </button>
      <HelpDialog isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
};
