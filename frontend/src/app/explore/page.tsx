'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useVisualization } from '@/contexts/VisualizationContext';
import { useAnimation } from '@/contexts/AnimationContext';
import { Header } from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { VisualizationCanvas } from '@/components/visualization/VisualizationCanvas';
import { ScrollytellingSection } from '@/components/scrollytelling/ScrollytellingSection';
import { PlaybackControls } from '@/components/controls/PlaybackControls';
import { ModuleContainer } from '@/components/modules/ModuleContainer';
import { cn } from '@/lib/design-system';
import { useScrollProgress } from '@/hooks/useScrollProgress';

// Scrollytelling sections configuration
const scrollytellingSections = [
  {
    id: 'introduction',
    title: 'Introduction to Transformers',
    content: `Transformers revolutionized natural language processing with their attention mechanism. 
    Unlike previous models that processed text sequentially, transformers can process all tokens simultaneously, 
    capturing complex relationships between words regardless of their position.`,
    visualizationStep: 0,
    backgroundColor: 'from-slate-50 to-blue-50',
  },
  {
    id: 'tokenization',
    title: 'Tokenization & Embedding',
    content: `The first step in processing text is breaking it down into tokens and converting them into 
    numerical representations called embeddings. Each token becomes a high-dimensional vector that captures 
    its semantic meaning. Positional encodings are added to give the model information about word order.`,
    visualizationStep: 1,
    backgroundColor: 'from-blue-50 to-indigo-50',
    modules: ['embedding'],
  },
  {
    id: 'attention',
    title: 'Multi-Head Attention',
    content: `The attention mechanism allows the model to weigh the importance of different tokens when 
    processing each token. Multi-head attention runs this process multiple times in parallel, allowing 
    the model to focus on different types of relationships simultaneously.`,
    visualizationStep: 2,
    backgroundColor: 'from-indigo-50 to-purple-50',
    modules: ['attention'],
  },
  {
    id: 'feedforward',
    title: 'Feed-Forward Networks',
    content: `Each transformer layer contains a feed-forward network that processes the output from the 
    attention mechanism. In modern architectures, this might be a Mixture of Experts (MoE) system, where 
    different experts specialize in different types of transformations.`,
    visualizationStep: 3,
    backgroundColor: 'from-purple-50 to-pink-50',
    modules: ['moe'],
  },
  {
    id: 'output',
    title: 'Output Generation',
    content: `The final layer produces logits for each possible token in the vocabulary. These are 
    converted to probabilities using softmax, and the model can either return the most likely token or 
    sample from the distribution for more creative outputs.`,
    visualizationStep: 4,
    backgroundColor: 'from-pink-50 to-amber-50',
    modules: ['output'],
  },
  {
    id: 'integration',
    title: 'Putting It All Together',
    content: `Modern transformer models stack multiple layers of these operations, with each layer 
    building increasingly sophisticated representations. The result is a powerful architecture capable 
    of understanding and generating human language with remarkable nuance.`,
    visualizationStep: 5,
    backgroundColor: 'from-amber-50 to-slate-50',
    modules: ['embedding', 'attention', 'moe', 'output'],
  },
];

export default function ExplorePage() {
  const { state: vizState, actions: vizActions } = useVisualization();
  const { state: animState, actions: animActions } = useAnimation();
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Custom hook for scroll progress
  const { scrollProgress, currentSectionIndex, isScrolling } = useScrollProgress({
    containerRef,
    sectionCount: scrollytellingSections.length,
    threshold: 0.5,
  });
  
  // Update visualization based on scroll progress
  useEffect(() => {
    if (currentSectionIndex >= 0 && currentSectionIndex < scrollytellingSections.length) {
      const section = scrollytellingSections[currentSectionIndex];
      
      // Update visualization step
      vizActions.setStep(section.visualizationStep);
      
      // Update active modules
      if (section.modules) {
        // Reset all modules first
        Object.keys(vizState.modules).forEach(module => {
          vizActions.updateModule(module as keyof typeof vizState.modules, { isActive: false });
        });
        
        // Activate specified modules
        section.modules.forEach(module => {
          vizActions.updateModule(module as keyof typeof vizState.modules, { isActive: true });
        });
      }
      
      // Update animation timeline
      animActions.seek((scrollProgress / 100) * animState.duration);
    }
  }, [currentSectionIndex, scrollProgress, vizActions, animActions, animState.duration, vizState.modules]);
  
  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case ' ':
          e.preventDefault();
          if (animState.isPlaying) {
            animActions.pause();
          } else {
            animActions.play();
          }
          break;
        case 'ArrowRight':
          if (currentSectionIndex < scrollytellingSections.length - 1) {
            const nextSection = containerRef.current?.children[currentSectionIndex + 1] as HTMLElement;
            nextSection?.scrollIntoView({ behavior: 'smooth' });
          }
          break;
        case 'ArrowLeft':
          if (currentSectionIndex > 0) {
            const prevSection = containerRef.current?.children[currentSectionIndex - 1] as HTMLElement;
            prevSection?.scrollIntoView({ behavior: 'smooth' });
          }
          break;
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [animState.isPlaying, currentSectionIndex, animActions]);
  
  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      {/* Fixed Header */}
      <Header />
      
      {/* Main Layout */}
      <div className="flex pt-16">
        {/* Sidebar */}
        <Sidebar />
        
        {/* Main Content */}
        <div className="flex-1 flex">
          {/* Scrollytelling Content */}
          <div className="flex-1 overflow-y-auto" ref={containerRef}>
            {scrollytellingSections.map((section, index) => (
              <ScrollytellingSection
                key={section.id}
                section={section}
                isActive={index === currentSectionIndex}
                progress={scrollProgress}
              />
            ))}
          </div>
          
          {/* Visualization Panel */}
          <div className="w-1/2 border-l border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 sticky top-16 h-[calc(100vh-4rem)]">
            <div className="h-full flex flex-col">
              {/* Visualization Canvas */}
              <div className="flex-1 relative">
                <VisualizationCanvas />
                
                {/* Module Overlays */}
                <div className="absolute inset-0 pointer-events-none">
                  {vizState.modules.embedding.isActive && (
                    <ModuleContainer type="embedding" className="pointer-events-auto" />
                  )}
                  {vizState.modules.attention.isActive && (
                    <ModuleContainer type="attention" className="pointer-events-auto" />
                  )}
                  {vizState.modules.moe.isActive && (
                    <ModuleContainer type="moe" className="pointer-events-auto" />
                  )}
                  {vizState.modules.output.isActive && (
                    <ModuleContainer type="output" className="pointer-events-auto" />
                  )}
                </div>
              </div>
              
              {/* Playback Controls */}
              <div className="border-t border-slate-200 dark:border-slate-700 p-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
                <PlaybackControls />
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Progress Indicator */}
      <div className="fixed right-4 top-1/2 -translate-y-1/2 z-50">
        <div className="flex flex-col gap-2">
          {scrollytellingSections.map((section, index) => (
            <button
              key={section.id}
              onClick={() => {
                const targetSection = containerRef.current?.children[index] as HTMLElement;
                targetSection?.scrollIntoView({ behavior: 'smooth' });
              }}
              className={cn(
                "w-2 h-2 rounded-full transition-all duration-300",
                index === currentSectionIndex
                  ? "bg-primary-500 w-8"
                  : "bg-slate-300 hover:bg-slate-400"
              )}
              title={section.title}
            />
          ))}
        </div>
      </div>
      
      {/* Keyboard Shortcuts Help */}
      <div className="fixed bottom-4 left-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm rounded-lg p-3 text-xs text-slate-600 dark:text-slate-400">
        <div className="font-semibold mb-1">Keyboard Shortcuts:</div>
        <div>Space: Play/Pause</div>
        <div>← →: Navigate sections</div>
      </div>
    </div>
  );
}