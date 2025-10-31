'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Layers, 
  Brain, 
  Network, 
  Zap,
  ChevronDown,
  ChevronRight,
  Settings,
  BookOpen,
  Activity,
  Eye
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/design-system';
import { useVisualization } from '@/contexts/VisualizationContext';

interface SidebarSection {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  items: {
    name: string;
    href: string;
    description?: string;
    badge?: string;
  }[];
}

const sidebarSections: SidebarSection[] = [
  {
    title: 'Learning Journey',
    icon: BookOpen,
    items: [
      {
        name: 'Interactive Explore',
        href: '/explore',
        description: 'Scroll-driven learning experience',
        badge: 'New',
      },
      {
        name: 'Quick Start',
        href: '/quickstart',
        description: 'Get started in minutes',
      },
    ],
  },
  {
    title: 'Components',
    icon: Layers,
    items: [
      {
        name: 'Embedding',
        href: '/modules/embedding',
        description: 'Token and position encoding',
      },
      {
        name: 'Attention',
        href: '/modules/attention',
        description: 'Multi-head attention mechanism',
      },
      {
        name: 'MoE',
        href: '/modules/moe',
        description: 'Mixture of Experts',
      },
      {
        name: 'Output Layer',
        href: '/modules/output',
        description: 'Prediction generation',
      },
    ],
  },
  {
    title: 'Tools',
    icon: Settings,
    items: [
      {
        name: 'Playground',
        href: '/playground',
        description: 'Experiment with custom inputs',
      },
      {
        name: 'Performance',
        href: '/performance',
        description: 'Benchmarking tools',
      },
      {
        name: 'API Reference',
        href: '/api-docs',
        description: 'Backend API documentation',
      },
    ],
  },
  {
    title: 'Resources',
    icon: BookOpen,
    items: [
      {
        name: 'Documentation',
        href: '/docs',
        description: 'Complete guide',
      },
      {
        name: 'Examples',
        href: '/examples',
        description: 'Code examples and tutorials',
      },
      {
        name: 'Research',
        href: '/research',
        description: 'Papers and citations',
      },
    ],
  },
];

export function Sidebar() {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['Learning Journey']));
  const pathname = usePathname();
  const { state: vizState } = useVisualization();

  const toggleSection = (sectionTitle: string) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionTitle)) {
        newSet.delete(sectionTitle);
      } else {
        newSet.add(sectionTitle);
      }
      return newSet;
    });
  };

  return (
    <aside className="w-64 bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 h-[calc(100vh-4rem)] overflow-y-auto">
      <div className="p-4">
        {/* Quick Status */}
        <div className="mb-6 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
              Current View
            </span>
            <Eye className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-sm font-semibold text-slate-900 dark:text-white capitalize">
            {vizState.currentView}
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Step {vizState.currentStep + 1} of {vizState.totalSteps || 0}
          </div>
        </div>

        {/* Navigation Sections */}
        <nav className="space-y-4">
          {sidebarSections.map((section) => (
            <div key={section.title} className="space-y-1">
              {/* Section Header */}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => toggleSection(section.title)}
                className="w-full justify-between px-2 py-1 h-auto hover:bg-slate-100 dark:hover:bg-slate-700"
              >
                <div className="flex items-center space-x-2">
                  <section.icon className="w-4 h-4 text-slate-500 dark:text-slate-400" />
                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                    {section.title}
                  </span>
                </div>
                {expandedSections.has(section.title) ? (
                  <ChevronDown className="w-4 h-4 text-slate-400" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-slate-400" />
                )}
              </Button>

              {/* Section Items */}
              {expandedSections.has(section.title) && (
                <div className="ml-6 space-y-0.5">
                  {section.items.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={cn(
                        "block px-3 py-2 rounded-md text-sm transition-colors duration-200 group",
                        pathname === item.href
                          ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 font-medium"
                          : "text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-300 dark:hover:text-white dark:hover:bg-slate-700"
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <span>{item.name}</span>
                        {item.badge && (
                          <span className="px-2 py-0.5 text-xs bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 rounded-full">
                            {item.badge}
                          </span>
                        )}
                      </div>
                      {item.description && (
                        <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 group-hover:text-slate-600 dark:group-hover:text-slate-300">
                          {item.description}
                        </div>
                      )}
                    </Link>
                  ))}
                </div>
              )}
            </div>
          ))}
        </nav>

        {/* Module Status */}
        <div className="mt-8 pt-6 border-t border-slate-200 dark:border-slate-700">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-slate-600 dark:text-slate-400">
              Active Modules
            </span>
            <Activity className="w-4 h-4 text-slate-400" />
          </div>
          <div className="space-y-2">
            {Object.entries(vizState.modules).map(([key, module]) => (
              <div
                key={key}
                className={cn(
                  "flex items-center space-x-2 px-2 py-1 rounded text-xs",
                  module.isActive
                    ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                    : "bg-slate-100 text-slate-500 dark:bg-slate-700 dark:text-slate-400"
                )}
              >
                <div
                  className={cn(
                    "w-2 h-2 rounded-full",
                    module.isActive ? "bg-green-500" : "bg-slate-400"
                  )}
                />
                <span className="capitalize">{key}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}