'use client';

import React from 'react';
import { cn } from '@/lib/design-system';
import { useScrollProgress } from '@/hooks/useScrollProgress';

interface ScrollytellingSectionProps {
  section: {
    id: string;
    title: string;
    content: string;
    visualizationStep: number;
    backgroundColor: string;
    modules?: string[];
  };
  isActive: boolean;
  progress: number;
}

export function ScrollytellingSection({ section, isActive, progress }: ScrollytellingSectionProps) {
  return (
    <section
      id={section.id}
      className={cn(
        "min-h-screen flex items-center justify-center p-8 transition-all duration-1000",
        `bg-gradient-to-br ${section.backgroundColor}`,
        isActive && "scale-100 opacity-100",
        !isActive && "scale-95 opacity-75"
      )}
    >
      <div className="max-w-4xl mx-auto">
        {/* Section Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <div className={cn(
              "w-1 h-8 rounded-full transition-all duration-500",
              isActive ? "bg-blue-500" : "bg-slate-300"
            )} />
            <h2 className={cn(
              "text-4xl font-bold transition-all duration-500",
              isActive ? "text-slate-900 dark:text-white" : "text-slate-500"
            )}>
              {section.title}
            </h2>
          </div>
          
          {/* Progress indicator */}
          <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 mb-8">
            <div
              className={cn(
                "h-2 rounded-full transition-all duration-300 bg-gradient-to-r from-blue-500 to-purple-600",
                isActive ? "w-full" : "w-0"
              )}
            />
          </div>
        </div>

        {/* Content */}
        <div className={cn(
          "prose prose-lg dark:prose-invert max-w-none transition-all duration-500",
          isActive ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"
        )}>
          <p className="text-lg leading-relaxed text-slate-700 dark:text-slate-300">
            {section.content}
          </p>
        </div>

        {/* Visual indicators */}
        <div className="mt-8 flex flex-wrap gap-2">
          {section.modules?.map((module) => (
            <span
              key={module}
              className={cn(
                "px-3 py-1 rounded-full text-sm font-medium transition-all duration-300",
                isActive
                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                  : "bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400"
              )}
            >
              {module}
            </span>
          ))}
        </div>

        {/* Step indicator */}
        <div className="mt-12 flex items-center justify-between">
          <div className="text-sm text-slate-500 dark:text-slate-400">
            Step {section.visualizationStep + 1}
          </div>
          {isActive && (
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              <span className="text-sm text-blue-600 dark:text-blue-400">Active</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}