import { motion } from 'framer-motion';
import { useExecutionStore } from '../store/executionStore';

export const SummaryPanel: React.FC = () => {
  const { data } = useExecutionStore();

  if (!data) return null;

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Output Summary</h3>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4 p-4 bg-gray-50 rounded-lg">
          <div>
            <p className="text-sm text-gray-600">Input Text</p>
            <p className="font-medium text-gray-900 mt-1">{data.inputText}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Token Count</p>
            <p className="font-medium text-gray-900 mt-1">{data.tokenCount} tokens</p>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Tokens</h4>
          <div className="flex flex-wrap gap-2">
            {data.tokens.map((token) => (
              <motion.span
                key={token.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: token.id * 0.05 }}
                className="px-3 py-1.5 bg-primary-100 text-primary-800 rounded-full text-sm font-medium"
                whileHover={{ scale: 1.05 }}
                title={`Token ID: ${token.id}`}
              >
                {token.text}
              </motion.span>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Output Probabilities</h4>
          <div className="space-y-2">
            {data.outputProbabilities.map((prob, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1 }}
                className="flex items-center gap-3"
              >
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium text-gray-700">{prob.token}</span>
                    <span className="text-sm text-gray-600">{(prob.probability * 100).toFixed(2)}%</span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${prob.probability * 100}%` }}
                      transition={{ duration: 0.5, delay: idx * 0.1 }}
                      className="h-full bg-gradient-to-r from-primary-500 to-primary-600 rounded-full"
                    />
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="pt-4 border-t border-gray-200">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Model Information</h4>
          <div className="text-sm text-gray-600 space-y-1">
            <p>Total Layers: {data.steps.length}</p>
            <p>Processing completed successfully</p>
          </div>
        </div>
      </div>
    </div>
  );
};
