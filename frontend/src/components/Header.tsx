'use client';

import { RotateCcw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/contexts/ThemeContext';
import { PerformanceSettings } from './PerformanceSettings';

interface HeaderProps {
  onReset?: () => void;
}

export const Header = ({ onReset }: HeaderProps) => {
  return (
    <header className="sticky top-0 z-50 backdrop-blur-lg bg-slate-900/80 border-b border-slate-700/50">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex items-center justify-between gap-4">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-bold text-white">
                Transformer{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400">
                  Visualizer
                </span>
              </h1>
              <p className="text-xs text-slate-400">稀疏MoE架构深度解析</p>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2 sm:gap-3">
            <PerformanceSettings />
            <ThemeToggle />
            {onReset && (
              <Button variant="secondary" size="sm" onClick={onReset} className="hidden sm:flex">
                <RotateCcw className="w-4 h-4" />
                重置
              </Button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
