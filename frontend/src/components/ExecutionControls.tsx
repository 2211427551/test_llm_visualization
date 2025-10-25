import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';
import { useExecutionStore } from '../store/executionStore';

export const ExecutionControls: React.FC = () => {
  const {
    data,
    currentStepIndex,
    isPlaying,
    playbackSpeed,
    setIsPlaying,
    setPlaybackSpeed,
    nextStep,
    previousStep,
    setCurrentStep,
    reset,
  } = useExecutionStore();

  if (!data) return null;

  const totalSteps = data.steps.length;
  const isFirstStep = currentStepIndex === 0;
  const isLastStep = currentStepIndex === totalSteps - 1;

  const handlePlayPause = () => {
    if (isLastStep && !isPlaying) {
      setCurrentStep(0);
    }
    setIsPlaying(!isPlaying);
  };

  const speedOptions = [0.5, 1, 1.5, 2];

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Execution Controls</h3>
        <button
          onClick={reset}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md transition-colors"
          aria-label="Reset execution"
        >
          <RotateCcw size={16} />
          Reset
        </button>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-center gap-2">
          <button
            onClick={previousStep}
            disabled={isFirstStep}
            className={`p-3 rounded-lg transition-all ${
              isFirstStep
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-primary-100 text-primary-700 hover:bg-primary-200 active:scale-95'
            }`}
            aria-label="Previous step (Left arrow)"
            title="Previous step (←)"
          >
            <SkipBack size={20} />
          </button>

          <button
            onClick={handlePlayPause}
            className="p-4 bg-primary-600 text-white rounded-lg hover:bg-primary-700 active:scale-95 transition-all"
            aria-label={isPlaying ? 'Pause playback (Space)' : 'Play playback (Space)'}
            title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
          >
            {isPlaying ? <Pause size={24} /> : <Play size={24} />}
          </button>

          <button
            onClick={nextStep}
            disabled={isLastStep}
            className={`p-3 rounded-lg transition-all ${
              isLastStep
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-primary-100 text-primary-700 hover:bg-primary-200 active:scale-95'
            }`}
            aria-label="Next step (Right arrow)"
            title="Next step (→)"
          >
            <SkipForward size={20} />
          </button>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-gray-600">
            <span>Step {currentStepIndex + 1} of {totalSteps}</span>
            <span className="text-xs">{data.steps[currentStepIndex]?.description}</span>
          </div>

          <div className="relative pt-1">
            <input
              type="range"
              min="0"
              max={totalSteps - 1}
              value={currentStepIndex}
              onChange={(e) => setCurrentStep(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              aria-label="Step progress"
            />
            <div 
              className="absolute top-1 left-0 h-2 bg-primary-500 rounded-lg pointer-events-none transition-all duration-200"
              style={{ width: `${((currentStepIndex + 1) / totalSteps) * 100}%` }}
            />
          </div>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-gray-200">
          <span className="text-sm text-gray-600">Playback Speed:</span>
          <div className="flex gap-2">
            {speedOptions.map((speed) => (
              <button
                key={speed}
                onClick={() => setPlaybackSpeed(speed)}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  playbackSpeed === speed
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                aria-label={`Set speed to ${speed}x`}
              >
                {speed}x
              </button>
            ))}
          </div>
        </div>

        <div className="text-xs text-gray-500 pt-2 border-t border-gray-100">
          <p className="font-medium mb-1">Keyboard Shortcuts:</p>
          <div className="grid grid-cols-2 gap-1">
            <span>← / → : Previous / Next</span>
            <span>Space: Play / Pause</span>
            <span>Home: First step</span>
            <span>End: Last step</span>
          </div>
        </div>
      </div>
    </div>
  );
};
