import { useState, useCallback, useEffect } from 'react';

export type PerformanceMode = 'high' | 'balanced' | 'low';

interface PerformanceSettings {
  animationDuration: number;
  maxElements: number;
  enableTransitions: boolean;
  enableShadows: boolean;
  enableBlur: boolean;
  particleCount: number;
}

const performancePresets: Record<PerformanceMode, PerformanceSettings> = {
  high: {
    animationDuration: 1000,
    maxElements: 10000,
    enableTransitions: true,
    enableShadows: true,
    enableBlur: true,
    particleCount: 100,
  },
  balanced: {
    animationDuration: 600,
    maxElements: 5000,
    enableTransitions: true,
    enableShadows: true,
    enableBlur: false,
    particleCount: 50,
  },
  low: {
    animationDuration: 300,
    maxElements: 1000,
    enableTransitions: false,
    enableShadows: false,
    enableBlur: false,
    particleCount: 20,
  },
};

/**
 * Hook for managing performance settings
 * Automatically detects device capabilities and provides appropriate defaults
 */
export function usePerformanceMode(initialMode?: PerformanceMode) {
  const [mode, setMode] = useState<PerformanceMode>(() => {
    if (initialMode) return initialMode;
    
    // Auto-detect performance mode based on device
    if (typeof window === 'undefined') return 'balanced';
    
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent
    );
    const hasLimitedRAM = (navigator as any).deviceMemory && (navigator as any).deviceMemory < 4;
    
    if (isMobile || hasLimitedRAM) {
      return 'low';
    }
    
    return 'balanced';
  });

  const settings = performancePresets[mode];

  const updateMode = useCallback((newMode: PerformanceMode) => {
    setMode(newMode);
    // Save to localStorage for persistence
    if (typeof window !== 'undefined') {
      localStorage.setItem('performanceMode', newMode);
    }
  }, []);

  // Load from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined' && !initialMode) {
      const saved = localStorage.getItem('performanceMode') as PerformanceMode;
      if (saved && performancePresets[saved]) {
        setMode(saved);
      }
    }
  }, [initialMode]);

  return {
    mode,
    setMode: updateMode,
    settings,
  };
}

/**
 * Hook for monitoring performance metrics
 */
export function usePerformanceMonitor() {
  const [fps, setFps] = useState(60);
  const [memoryUsage, setMemoryUsage] = useState(0);

  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationId: number;

    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();

      if (currentTime >= lastTime + 1000) {
        setFps(Math.round((frameCount * 1000) / (currentTime - lastTime)));
        frameCount = 0;
        lastTime = currentTime;

        // Measure memory if available
        if ((performance as any).memory) {
          const memory = (performance as any).memory;
          setMemoryUsage(
            Math.round((memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100)
          );
        }
      }

      animationId = requestAnimationFrame(measureFPS);
    };

    animationId = requestAnimationFrame(measureFPS);

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, []);

  return { fps, memoryUsage };
}
