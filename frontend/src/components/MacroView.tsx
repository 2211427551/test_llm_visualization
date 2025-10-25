import { motion } from 'framer-motion';
import { useExecutionStore } from '../store/executionStore';
import { AlertTriangle } from 'lucide-react';

export const MacroView: React.FC = () => {
  const { data, currentStepIndex } = useExecutionStore();

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <p className="text-gray-500">Run a model to see the macro view</p>
      </div>
    );
  }

  const currentStep = data.steps[currentStepIndex];
  const currentLayerId = currentStep.layerData.layerId;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Macro View - Model Architecture</h3>
        {data.truncated && (
          <div className="flex items-center gap-2 text-sm text-yellow-600">
            <AlertTriangle size={16} />
            <span>Some tensors truncated</span>
          </div>
        )}
      </div>

      <div className="space-y-3">
        {data.steps.map((step, idx) => {
          const isActive = idx === currentStepIndex;
          const isPast = idx < currentStepIndex;
          const layer = step.layerData;

          return (
            <motion.div
              key={step.stepIndex}
              initial={{ opacity: 0, x: -20 }}
              animate={{ 
                opacity: 1, 
                x: 0,
                scale: isActive ? 1.02 : 1,
              }}
              transition={{ 
                duration: 0.3,
                delay: idx * 0.05,
              }}
              className={`relative p-4 rounded-lg border-2 transition-all ${
                isActive
                  ? 'border-primary-500 bg-primary-50 shadow-md'
                  : isPast
                  ? 'border-green-300 bg-green-50'
                  : 'border-gray-200 bg-gray-50'
              }`}
            >
              {isActive && (
                <motion.div
                  layoutId="active-layer"
                  className="absolute inset-0 bg-primary-100 rounded-lg"
                  style={{ zIndex: -1 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />
              )}

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold ${
                    isActive ? 'bg-primary-600 text-white' : isPast ? 'bg-green-600 text-white' : 'bg-gray-300 text-gray-600'
                  }`}>
                    {idx + 1}
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-800">{layer.layerName}</h4>
                    <p className="text-sm text-gray-600">{step.description}</p>
                  </div>
                </div>

                <div className="text-right text-sm">
                  <p className="text-gray-600">
                    Input: [{layer.inputShape.join(', ')}]
                  </p>
                  <p className="text-gray-600">
                    Output: [{layer.outputShape.join(', ')}]
                  </p>
                  {layer.truncated && (
                    <p className="text-yellow-600 text-xs flex items-center justify-end gap-1 mt-1">
                      <AlertTriangle size={12} />
                      Truncated
                    </p>
                  )}
                </div>
              </div>

              {isActive && idx < data.steps.length - 1 && (
                <motion.div
                  className="absolute -bottom-3 left-1/2 transform -translate-x-1/2"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <motion.div
                    className="w-1 h-6 bg-primary-500"
                    animate={{
                      scaleY: [1, 1.5, 1],
                    }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "easeInOut",
                    }}
                  />
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>

      {data.warnings && data.warnings.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg"
        >
          <h4 className="flex items-center gap-2 text-sm font-semibold text-yellow-800 mb-2">
            <AlertTriangle size={16} />
            Warnings
          </h4>
          <ul className="text-sm text-yellow-700 space-y-1">
            {data.warnings.map((warning, idx) => (
              <li key={idx}>• {warning}</li>
            ))}
          </ul>
        </motion.div>
      )}
    </div>
  );
};
