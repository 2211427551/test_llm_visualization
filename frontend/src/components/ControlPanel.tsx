'use client';

import { useEffect } from 'react';
import { useVisualizationStore } from '@/store/visualizationStore';
import { Card } from './ui';
import { Play, Pause, SkipBack, SkipForward, Settings } from 'lucide-react';

export default function ControlPanel() {
  const {
    isInitialized,
    isPlaying,
    currentStep,
    totalSteps,
    playbackSpeed,
    togglePlayback,
    nextStep,
    prevStep,
    goToStep,
    setPlaybackSpeed,
    currentStepData,
  } = useVisualizationStore();

  useEffect(() => {
    if (!isPlaying || !isInitialized) return;

    const interval = setInterval(() => {
      if (currentStep < totalSteps - 1) {
        nextStep();
      } else {
        togglePlayback();
      }
    }, 1000 / playbackSpeed);

    return () => clearInterval(interval);
  }, [isPlaying, currentStep, totalSteps, playbackSpeed, isInitialized, nextStep, togglePlayback]);

  if (!isInitialized) {
    return (
      <Card className="mb-6">
        <p className="text-slate-400 text-center py-4">请先输入文本并初始化计算</p>
      </Card>
    );
  }

  return (
    <Card className="mb-6">
      <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Settings className="w-5 h-5 text-purple-400" />
        播放控制
      </h3>
      
      {/* Current Step Info */}
      <div className="mb-4">
        <div className="flex justify-between text-sm text-slate-400 mb-2">
          <span>步骤 {currentStep + 1} / {totalSteps}</span>
          {currentStepData && (
            <span>Layer {currentStepData.layer_index + 1} - {currentStepData.step_type}</span>
          )}
        </div>
        
        {/* Progress Bar */}
        <div className="relative h-2 bg-slate-700 rounded-full overflow-hidden">
          <div 
            className="absolute h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
            style={{ width: `${((currentStep + 1) / totalSteps) * 100}%` }}
          />
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center justify-center gap-3 mb-6">
        <button
          className="p-3 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed border border-slate-600"
          onClick={prevStep}
          disabled={currentStep === 0}
          title="上一步"
        >
          <SkipBack className="w-5 h-5 text-slate-300" />
        </button>

        <button
          className="p-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white rounded-full transition-all duration-200 shadow-lg shadow-purple-500/30"
          onClick={togglePlayback}
          title={isPlaying ? '暂停' : '播放'}
        >
          {isPlaying ? (
            <Pause className="w-6 h-6" />
          ) : (
            <Play className="w-6 h-6 ml-0.5" />
          )}
        </button>

        <button
          className="p-3 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed border border-slate-600"
          onClick={nextStep}
          disabled={currentStep === totalSteps - 1}
          title="下一步"
        >
          <SkipForward className="w-5 h-5 text-slate-300" />
        </button>
      </div>

      {/* Speed Control */}
      <div className="mb-4">
        <label className="text-sm text-slate-300 mb-2 block">播放速度</label>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 w-8">0.5x</span>
          <input 
            type="range" 
            min="0.5" 
            max="3" 
            step="0.5"
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            className="flex-1"
          />
          <span className="text-xs text-slate-500 w-8">3x</span>
        </div>
        <div className="text-center mt-2">
          <span className="inline-block bg-purple-500/20 text-purple-300 px-3 py-1 rounded-full text-sm border border-purple-500/30">
            {playbackSpeed}x
          </span>
        </div>
      </div>

      {/* Step Selector */}
      <div>
        <label className="text-sm text-slate-300 mb-2 block">跳转到步骤</label>
        <select 
          className="w-full bg-slate-900/50 border border-slate-600 rounded-lg px-3 py-2 text-white focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 outline-none"
          value={currentStep}
          onChange={(e) => goToStep(parseInt(e.target.value))}
        >
          {Array.from({ length: totalSteps }, (_, i) => (
            <option key={i} value={i}>
              步骤 {i + 1}
            </option>
          ))}
        </select>
      </div>
    </Card>
  );
}
