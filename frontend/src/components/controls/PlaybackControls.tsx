'use client';

import React from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  SkipBack, 
  SkipForward, 
  RotateCcw,
  Volume2,
  VolumeX,
  Settings
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { useAnimation } from '@/contexts/AnimationContext';
import { useVisualization } from '@/contexts/VisualizationContext';
import { cn } from '@/lib/design-system';

export function PlaybackControls() {
  const { state: animState, actions: animActions } = useAnimation();
  const { state: vizState, actions: vizActions } = useVisualization();

  const handleProgressChange = (value: number[]) => {
    const progress = value[0];
    const time = (progress / 100) * animState.duration;
    animActions.seek(time);
  };

  const handleSpeedChange = (value: number[]) => {
    animActions.setPlaybackSpeed(value[0]);
  };

  const stepForward = () => {
    if (vizState.currentStep < vizState.totalSteps - 1) {
      vizActions.nextStep();
    }
  };

  const stepBackward = () => {
    if (vizState.currentStep > 0) {
      vizActions.previousStep();
    }
  };

  return (
    <div className="space-y-4">
      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
          <span>Progress</span>
          <span>{Math.round(animState.progress)}%</span>
        </div>
        <Slider
          value={[animState.progress]}
          onValueChange={handleProgressChange}
          max={100}
          step={1}
          className="w-full"
        />
      </div>

      {/* Main Controls */}
      <div className="flex items-center justify-center space-x-2">
        {/* Reset */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            animActions.stop();
            vizActions.setStep(0);
          }}
          className="p-2"
        >
          <RotateCcw className="w-4 h-4" />
        </Button>

        {/* Step Backward */}
        <Button
          variant="outline"
          size="sm"
          onClick={stepBackward}
          disabled={vizState.currentStep === 0}
          className="p-2"
        >
          <SkipBack className="w-4 h-4" />
        </Button>

        {/* Play/Pause */}
        <Button
          variant={animState.isPlaying ? "default" : "default"}
          size="sm"
          onClick={() => {
            if (animState.isPlaying) {
              animActions.pause();
            } else {
              animActions.play();
            }
          }}
          className="p-3"
        >
          {animState.isPlaying ? (
            <Pause className="w-5 h-5" />
          ) : (
            <Play className="w-5 h-5" />
          )}
        </Button>

        {/* Step Forward */}
        <Button
          variant="outline"
          size="sm"
          onClick={stepForward}
          disabled={vizState.currentStep >= vizState.totalSteps - 1}
          className="p-2"
        >
          <SkipForward className="w-4 h-4" />
        </Button>

        {/* Stop */}
        <Button
          variant="outline"
          size="sm"
          onClick={animActions.stop}
          className="p-2"
        >
          <Square className="w-4 h-4" />
        </Button>
      </div>

      {/* Secondary Controls */}
      <div className="flex items-center justify-between">
        {/* Speed Control */}
        <div className="flex items-center space-x-2 flex-1">
          <span className="text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">
            Speed
          </span>
          <Slider
            value={[animState.playbackSpeed]}
            onValueChange={handleSpeedChange}
            min={0.25}
            max={2}
            step={0.25}
            className="w-20"
          />
          <span className="text-xs text-slate-500 dark:text-slate-400 whitespace-nowrap">
            {animState.playbackSpeed}x
          </span>
        </div>

        {/* Loop Control */}
        <Button
          variant={animState.isLooping ? "default" : "outline"}
          size="sm"
          onClick={animActions.toggleLoop}
          className="text-xs px-2 py-1"
        >
          Loop
        </Button>

        {/* Reverse Control */}
        <Button
          variant={animState.isReversed ? "default" : "outline"}
          size="sm"
          onClick={animActions.toggleReverse}
          className="text-xs px-2 py-1"
        >
          Reverse
        </Button>
      </div>

      {/* Time Display */}
      <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 pt-2 border-t border-slate-200 dark:border-slate-700">
        <div>
          Current: {Math.floor(animState.currentTime)}s
        </div>
        <div>
          Duration: {Math.floor(animState.duration)}s
        </div>
      </div>

      {/* Keyframes */}
      {animState.keyframes.length > 0 && (
        <div className="pt-2 border-t border-slate-200 dark:border-slate-700">
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">
            Keyframes
          </div>
          <div className="flex space-x-1">
            {animState.keyframes.map((keyframe, index) => (
              <Button
                key={keyframe.id}
                variant="outline"
                size="sm"
                onClick={() => animActions.seek(keyframe.time)}
                className={cn(
                  "px-2 py-1 text-xs",
                  animState.currentTime >= keyframe.time && "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300"
                )}
                title={keyframe.label}
              >
                {index + 1}
              </Button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}