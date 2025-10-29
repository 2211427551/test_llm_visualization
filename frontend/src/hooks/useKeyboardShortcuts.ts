import { useEffect, useCallback } from 'react';

type ShortcutHandler = (event: KeyboardEvent) => void;

interface Shortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  handler: ShortcutHandler;
  description?: string;
}

/**
 * Hook for managing keyboard shortcuts
 * Supports various modifier keys and provides automatic cleanup
 */
export function useKeyboardShortcuts(shortcuts: Shortcut[], enabled = true) {
  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      for (const shortcut of shortcuts) {
        const keyMatches = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatches = shortcut.ctrl === undefined || event.ctrlKey === shortcut.ctrl;
        const shiftMatches = shortcut.shift === undefined || event.shiftKey === shortcut.shift;
        const altMatches = shortcut.alt === undefined || event.altKey === shortcut.alt;

        if (keyMatches && ctrlMatches && shiftMatches && altMatches) {
          event.preventDefault();
          shortcut.handler(event);
          break;
        }
      }
    },
    [shortcuts, enabled]
  );

  useEffect(() => {
    if (!enabled) return;

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown, enabled]);
}

/**
 * Common keyboard shortcuts for visualization controls
 */
export function useVisualizationShortcuts(
  onPlay?: () => void,
  onPause?: () => void,
  onNext?: () => void,
  onPrev?: () => void,
  onReset?: () => void
) {
  const shortcuts: Shortcut[] = [
    {
      key: ' ',
      handler: () => {
        if (onPlay) onPlay();
        if (onPause) onPause();
      },
      description: 'Play/Pause',
    },
    {
      key: 'ArrowRight',
      handler: () => onNext?.(),
      description: 'Next step',
    },
    {
      key: 'ArrowLeft',
      handler: () => onPrev?.(),
      description: 'Previous step',
    },
    {
      key: 'r',
      handler: () => onReset?.(),
      description: 'Reset',
    },
    {
      key: 'Escape',
      handler: () => onPause?.(),
      description: 'Pause',
    },
  ];

  useKeyboardShortcuts(shortcuts);

  return shortcuts;
}
