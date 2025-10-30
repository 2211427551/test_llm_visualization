'use client';

import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';
import { usePlaybackStore } from '@/stores/playback-store';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card, CardContent } from '@/components/ui/card';
import { useEffect } from 'react';

export function PlaybackControls() {
  const { 
    isPlaying, 
    currentStep, 
    totalSteps,
    speed,
    progress,
    steps,
    play,
    pause,
    nextStep,
    prevStep,
    reset,
    setSpeed,
    goToStep,
    updateProgress,
  } = usePlaybackStore();
  
  // Animation loop
  useEffect(() => {
    if (!isPlaying) return;
    
    let lastTime = Date.now();
    let animationFrameId: number;
    
    const loop = () => {
      const currentTime = Date.now();
      const delta = (currentTime - lastTime) / 1000;
      lastTime = currentTime;
      
      updateProgress(delta * 0.2); // Adjust speed factor
      
      if (isPlaying) {
        animationFrameId = requestAnimationFrame(loop);
      }
    };
    
    animationFrameId = requestAnimationFrame(loop);
    
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isPlaying, updateProgress]);
  
  const currentStepInfo = steps[currentStep];
  const progressPercent = ((currentStep + progress) / totalSteps) * 100;
  
  return (
    <Card className="bg-slate-800 border-slate-700">
      <CardContent className="p-6 space-y-6">
        {/* Step info */}
        {currentStepInfo && (
          <div className="text-center space-y-1">
            <div className="text-sm text-slate-400">
              Step {currentStep + 1} / {totalSteps}
            </div>
            <div className="text-white font-medium">
              {currentStepInfo.name}
            </div>
            <div className="text-xs text-slate-500">
              {currentStepInfo.description}
            </div>
          </div>
        )}
        
        {/* Main controls */}
        <div className="flex items-center justify-center gap-3">
          <Button
            size="icon"
            variant="ghost"
            onClick={reset}
            className="hover:bg-slate-700"
            title="Reset to beginning"
          >
            <RotateCcw className="w-5 h-5" />
          </Button>
          
          <Button
            size="icon"
            variant="ghost"
            onClick={prevStep}
            disabled={currentStep === 0}
            className="hover:bg-slate-700"
            title="Previous step"
          >
            <SkipBack className="w-5 h-5" />
          </Button>
          
          <Button
            size="icon"
            onClick={isPlaying ? pause : play}
            className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <Pause className="w-8 h-8" />
            ) : (
              <Play className="w-8 h-8 ml-1" />
            )}
          </Button>
          
          <Button
            size="icon"
            variant="ghost"
            onClick={nextStep}
            disabled={currentStep === totalSteps - 1}
            className="hover:bg-slate-700"
            title="Next step"
          >
            <SkipForward className="w-5 h-5" />
          </Button>
        </div>
        
        {/* Progress bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-slate-400">
            <span>Progress</span>
            <span>{progressPercent.toFixed(0)}%</span>
          </div>
          
          <div className="relative h-2 bg-slate-700 rounded-full overflow-hidden">
            <div 
              className="absolute inset-y-0 left-0 bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-100"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          
          {/* Step markers */}
          {totalSteps > 0 && (
            <div className="relative h-6">
              {Array.from({ length: totalSteps }).map((_, i) => (
                <button
                  key={i}
                  onClick={() => goToStep(i)}
                  className={`absolute w-2 h-2 rounded-full transition-all ${
                    i === currentStep 
                      ? 'bg-pink-500 scale-150' 
                      : i < currentStep 
                      ? 'bg-purple-500' 
                      : 'bg-slate-600'
                  }`}
                  style={{ left: `${(i / (totalSteps - 1)) * 100}%` }}
                  title={steps[i]?.name}
                />
              ))}
            </div>
          )}
        </div>
        
        {/* Speed control */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-300">Speed</span>
            <span className="text-white font-mono">{speed.toFixed(1)}x</span>
          </div>
          
          <Slider
            value={[speed]}
            onValueChange={([value]) => setSpeed(value)}
            min={0.25}
            max={3}
            step={0.25}
            className="cursor-pointer"
          />
          
          <div className="flex justify-between text-xs text-slate-500">
            <span>0.25x</span>
            <span>3x</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
