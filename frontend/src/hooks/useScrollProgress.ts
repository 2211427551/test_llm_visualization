'use client';

import { useState, useEffect, useCallback, useRef, RefObject } from 'react';

interface UseScrollProgressOptions {
  containerRef: RefObject<HTMLDivElement | null>;
  sectionCount: number;
  threshold?: number;
}

interface ScrollProgressReturn {
  scrollProgress: number;
  currentSectionIndex: number;
  isScrolling: boolean;
  scrollToSection: (index: number) => void;
}

export function useScrollProgress({
  containerRef,
  sectionCount,
  threshold = 0.5,
}: UseScrollProgressOptions): ScrollProgressReturn {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [currentSectionIndex, setCurrentSectionIndex] = useState(0);
  const [isScrolling, setIsScrolling] = useState(false);
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Calculate scroll progress and current section
  const updateScrollProgress = useCallback(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const scrollTop = container.scrollTop;
    const scrollHeight = container.scrollHeight - container.clientHeight;
    
    // Calculate overall progress (0-100)
    const progress = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
    setScrollProgress(Math.min(100, Math.max(0, progress)));

    // Calculate current section based on scroll position
    const sectionHeight = scrollHeight / sectionCount;
    const currentSection = Math.min(
      Math.floor(scrollTop / sectionHeight),
      sectionCount - 1
    );
    
    // Check if we're in the threshold area for the next section
    const sectionProgress = (scrollTop % sectionHeight) / sectionHeight;
    if (sectionProgress > threshold && currentSection < sectionCount - 1) {
      setCurrentSectionIndex(currentSection + 1);
    } else {
      setCurrentSectionIndex(currentSection);
    }
  }, [containerRef, sectionCount, threshold]);

  // Handle scroll events
  const handleScroll = useCallback(() => {
    setIsScrolling(true);
    updateScrollProgress();
    
    // Clear existing timeout
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current);
    }
    
    // Set scrolling to false after scroll ends
    scrollTimeoutRef.current = setTimeout(() => {
      setIsScrolling(false);
    }, 150);
  }, [updateScrollProgress]);

  // Scroll to specific section
  const scrollToSection = useCallback((index: number) => {
    if (!containerRef.current || index < 0 || index >= sectionCount) return;

    const container = containerRef.current;
    const sectionHeight = (container.scrollHeight - container.clientHeight) / sectionCount;
    const targetScrollTop = index * sectionHeight;
    
    container.scrollTo({
      top: targetScrollTop,
      behavior: 'smooth'
    });
  }, [containerRef, sectionCount]);

  // Set up scroll listener
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('scroll', handleScroll, { passive: true });
    
    // Initial calculation
    updateScrollProgress();

    return () => {
      container.removeEventListener('scroll', handleScroll);
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current);
      }
    };
  }, [containerRef, handleScroll, updateScrollProgress]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      updateScrollProgress();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [updateScrollProgress]);

  return {
    scrollProgress,
    currentSectionIndex,
    isScrolling,
    scrollToSection,
  };
}