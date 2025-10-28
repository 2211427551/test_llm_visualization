'use client';

import { useEffect } from 'react';
import { useVisualizationStore } from '@/store/visualizationStore';

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

  // Auto-play functionality
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
      <div className="w-full bg-gray-100 rounded-lg shadow-md p-6 mb-6">
        <p className="text-gray-500 text-center">请先输入文本并初始化计算</p>
      </div>
    );
  }

  return (
    <div className="w-full bg-white rounded-lg shadow-md p-6 mb-6">
      <div className="flex flex-col space-y-4">
        {/* Current Step Info */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">
              步骤 {currentStep + 1} / {totalSteps}
            </h3>
            {currentStepData && (
              <p className="text-sm text-gray-600">
                第 {currentStepData.layer_index + 1} 层 - {currentStepData.step_type}
              </p>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${((currentStep + 1) / totalSteps) * 100}%` }}
          />
        </div>

        {/* Control Buttons */}
        <div className="flex items-center justify-center space-x-4">
          <button
            className="p-3 bg-gray-200 hover:bg-gray-300 rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={prevStep}
            disabled={currentStep === 0}
            title="上一步"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
            </svg>
          </button>

          <button
            className="p-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition duration-200"
            onClick={togglePlayback}
            title={isPlaying ? '暂停' : '播放'}
          >
            {isPlaying ? (
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            )}
          </button>

          <button
            className="p-3 bg-gray-200 hover:bg-gray-300 rounded-lg transition duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={nextStep}
            disabled={currentStep === totalSteps - 1}
            title="下一步"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
            </svg>
          </button>
        </div>

        {/* Speed Control */}
        <div className="flex items-center space-x-4">
          <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
            速度: {playbackSpeed}x
          </label>
          <input
            type="range"
            min="0.5"
            max="3"
            step="0.5"
            value={playbackSpeed}
            onChange={(e) => setPlaybackSpeed(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Layer Selector */}
        <div className="flex items-center space-x-4">
          <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
            跳转到步骤:
          </label>
          <select
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
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
      </div>
    </div>
  );
}
