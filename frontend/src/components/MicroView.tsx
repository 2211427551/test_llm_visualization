import { motion } from 'framer-motion';
import { useExecutionStore } from '../store/executionStore';
import { AttentionInspector } from './AttentionInspector';
import { MoEInspector } from './MoEInspector';
import { Breadcrumb } from './Breadcrumb';

export const MicroView: React.FC = () => {
  const { data, currentStepIndex } = useExecutionStore();

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <p className="text-gray-500">Run a model to see the micro view</p>
      </div>
    );
  }

  const currentStep = data.steps[currentStepIndex];
  const layer = currentStep.layerData;
  const tokens = data.tokens.map(t => t.text);

  const renderMatrix = (matrix: number[][] | undefined, label: string) => {
    if (!matrix || matrix.length === 0) return null;

    const maxDisplay = 8;
    const displayRows = matrix.slice(0, maxDisplay);
    const hasMore = matrix.length > maxDisplay;

    return (
      <div className="space-y-2">
        <h5 className="text-sm font-semibold text-gray-700">{label}</h5>
        <div className="bg-gray-50 p-3 rounded-lg overflow-x-auto">
          <div className="font-mono text-xs space-y-1">
            {displayRows.map((row, rowIdx) => (
              <motion.div
                key={rowIdx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: rowIdx * 0.05 }}
                className="flex gap-2"
              >
                {row.slice(0, 10).map((value, colIdx) => (
                  <motion.span
                    key={colIdx}
                    className={`inline-block w-16 text-center px-2 py-1 rounded ${
                      Math.abs(value) > 0.5 ? 'bg-primary-200 text-primary-900' : 'bg-gray-200 text-gray-700'
                    }`}
                    whileHover={{ scale: 1.1, zIndex: 10 }}
                    title={`[${rowIdx}, ${colIdx}]: ${value.toFixed(4)}`}
                  >
                    {value.toFixed(3)}
                  </motion.span>
                ))}
                {row.length > 10 && <span className="text-gray-400">... +{row.length - 10} more</span>}
              </motion.div>
            ))}
            {hasMore && (
              <div className="text-gray-400 text-center py-1">
                ... +{matrix.length - maxDisplay} more rows
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderLayerContent = () => {
    if (layer.layerType === 'attention' && layer.attentionData) {
      return <AttentionInspector data={layer.attentionData} tokens={tokens} />;
    }

    if (layer.layerType === 'moe' && layer.moeData) {
      return <MoEInspector data={layer.moeData} tokens={tokens} />;
    }

    return (
      <>
        {renderMatrix(layer.activations, 'Activations')}
        {renderMatrix(layer.weights, 'Weights')}
      </>
    );
  };

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <motion.div
        key={currentStepIndex}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="mb-4">
          <Breadcrumb items={['Model', `Step ${currentStepIndex + 1}`, layer.layerName]} />
        </div>

        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Micro View - Layer Details
        </h3>

        <div className="space-y-4">
          <div className="bg-primary-50 border border-primary-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-lg font-bold text-primary-900">{layer.layerName}</h4>
              {layer.layerType && (
                <span className="px-3 py-1 bg-primary-600 text-white text-xs font-semibold rounded-full uppercase">
                  {layer.layerType}
                </span>
              )}
            </div>
            <p className="text-sm text-primary-700 mb-3">{currentStep.description}</p>
            
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-semibold text-gray-700">Layer ID:</span>
                <span className="ml-2 text-gray-600">{layer.layerId}</span>
              </div>
              <div>
                <span className="font-semibold text-gray-700">Step:</span>
                <span className="ml-2 text-gray-600">{currentStepIndex + 1} / {data.steps.length}</span>
              </div>
              <div>
                <span className="font-semibold text-gray-700">Input Shape:</span>
                <span className="ml-2 text-gray-600">[{layer.inputShape.join(', ')}]</span>
              </div>
              <div>
                <span className="font-semibold text-gray-700">Output Shape:</span>
                <span className="ml-2 text-gray-600">[{layer.outputShape.join(', ')}]</span>
              </div>
            </div>
          </div>

          {renderLayerContent()}

          {layer.truncated && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="bg-yellow-50 border border-yellow-200 rounded-lg p-3"
            >
              <p className="text-sm text-yellow-800">
                ⚠️ This layer's data has been truncated by the backend to reduce payload size.
              </p>
            </motion.div>
          )}
        </div>
      </motion.div>
    </div>
  );
};
