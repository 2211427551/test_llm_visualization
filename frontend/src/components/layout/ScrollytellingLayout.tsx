'use client';

import { ReactNode, useEffect, useRef } from 'react';
import { useScroll, useTransform, motion } from 'framer-motion';

interface Section {
  id: string;
  title: string;
  content: ReactNode;
}

interface ScrollytellingLayoutProps {
  sections: Section[];
  visualization: (scrollProgress: number) => ReactNode;
}

export function ScrollytellingLayout({ sections, visualization }: ScrollytellingLayoutProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });
  
  return (
    <div ref={containerRef} className="relative">
      {/* Fixed visualization canvas */}
      <div className="sticky top-16 h-[calc(100vh-4rem)] flex items-center justify-center bg-slate-950">
        {visualization(scrollYProgress)}
      </div>
      
      {/* Scrolling content */}
      <div className="relative z-10 pointer-events-none">
        {sections.map((section, index) => (
          <ScrollSection
            key={section.id}
            step={index}
            totalSteps={sections.length}
            scrollProgress={scrollYProgress}
          >
            <div className="bg-white/95 dark:bg-slate-900/95 backdrop-blur-lg p-8 rounded-xl shadow-2xl max-w-md pointer-events-auto">
              <h2 className="text-2xl font-bold mb-4 text-slate-900 dark:text-white">
                {section.title}
              </h2>
              <div className="text-slate-700 dark:text-slate-300 space-y-3">
                {section.content}
              </div>
              <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700">
                <div className="text-sm text-slate-500">
                  Step {index + 1} of {sections.length}
                </div>
              </div>
            </div>
          </ScrollSection>
        ))}
        
        {/* Extra space at the end */}
        <div className="h-screen" />
      </div>
    </div>
  );
}

interface ScrollSectionProps {
  step: number;
  totalSteps: number;
  scrollProgress: number;
  children: ReactNode;
}

function ScrollSection({ step, totalSteps, scrollProgress, children }: ScrollSectionProps) {
  const stepProgress = 1 / totalSteps;
  const start = step * stepProgress;
  const middle = start + stepProgress / 2;
  const end = start + stepProgress;
  
  const opacity = useTransform(
    scrollProgress,
    [start - 0.1, start, middle, end, end + 0.1],
    [0, 1, 1, 1, 0]
  );
  
  const y = useTransform(
    scrollProgress,
    [start - 0.1, start, end, end + 0.1],
    [100, 0, 0, -100]
  );
  
  return (
    <motion.div 
      style={{ opacity, y }} 
      className="h-screen flex items-center justify-end pr-12"
    >
      {children}
    </motion.div>
  );
}
