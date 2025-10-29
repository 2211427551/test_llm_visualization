import { useEffect, useRef, useCallback } from 'react';
import React from 'react';
import * as d3 from 'd3';

/**
 * Custom hook for D3 visualization with automatic cleanup
 * Provides performance optimizations and lifecycle management
 */
export function useD3Visualization<T extends Element>(
  renderFn: (svg: d3.Selection<T, unknown, null, undefined>) => void | (() => void),
  dependencies: React.DependencyList = []
) {
  const ref = useRef<T>(null);
  const cleanupRef = useRef<(() => void) | void>(undefined);

  useEffect(() => {
    if (!ref.current) return;

    const svg = d3.select<T, unknown>(ref.current);
    cleanupRef.current = renderFn(svg);

    return () => {
      // Clean up D3 elements
      svg.selectAll('*').remove();
      if (typeof cleanupRef.current === 'function') {
        cleanupRef.current();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencies);

  return ref;
}

/**
 * Hook for managing D3 animations with requestAnimationFrame
 */
export function useD3Animation(
  animationFn: (progress: number) => void,
  duration: number,
  autoStart = false
) {
  const animationIdRef = useRef<number | undefined>(undefined);
  const startTimeRef = useRef<number | undefined>(undefined);

  const start = useCallback(() => {
    startTimeRef.current = performance.now();
    
    const animate = (currentTime: number) => {
      if (!startTimeRef.current) return;
      
      const elapsed = currentTime - startTimeRef.current;
      const progress = Math.min(elapsed / duration, 1);
      
      animationFn(progress);
      
      if (progress < 1) {
        animationIdRef.current = requestAnimationFrame(animate);
      }
    };
    
    animationIdRef.current = requestAnimationFrame(animate);
  }, [animationFn, duration]);

  const stop = useCallback(() => {
    if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current);
    }
  }, []);

  useEffect(() => {
    if (autoStart) {
      start();
    }
    return stop;
  }, [autoStart, start, stop]);

  return { start, stop };
}

/**
 * Hook for responsive SVG dimensions
 */
export function useResponsiveSVG(minWidth = 800, minHeight = 600) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = React.useState({ width: minWidth, height: minHeight });

  useEffect(() => {
    if (!containerRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setDimensions({
          width: Math.max(entry.contentRect.width, minWidth),
          height: Math.max(entry.contentRect.height, minHeight),
        });
      }
    });

    resizeObserver.observe(containerRef.current);

    return () => resizeObserver.disconnect();
  }, [minWidth, minHeight]);

  return { containerRef, ...dimensions };
}
