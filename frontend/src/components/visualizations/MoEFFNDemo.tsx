'use client';

import { useState } from 'react';
import { MoEFFNViz } from './MoEFFNViz';

export const MoEFFNDemo: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);

  // 示例配置
  const config = {
    n_experts: 8,
    top_k: 2,
    d_ff: 64,  // 4 * n_embd
    n_embd: 16,
  };

  const nTokens = 3;
  const nEmbd = config.n_embd;
  const nExperts = config.n_experts;
  const dFF = config.d_ff;

  // 生成随机输入数据
  const generateInputData = (): number[][] => {
    return Array(nTokens).fill(0).map(() =>
      Array(nEmbd).fill(0).map(() => (Math.random() - 0.5) * 2)
    );
  };

  // 生成随机权重
  const generateWeights = () => {
    return {
      ln_gamma: Array(nEmbd).fill(0).map(() => 0.8 + Math.random() * 0.4),
      ln_beta: Array(nEmbd).fill(0).map(() => (Math.random() - 0.5) * 0.2),
      gate_weights: Array(nEmbd).fill(0).map(() =>
        Array(nExperts).fill(0).map(() => (Math.random() - 0.5) * 0.5)
      ),
      experts: Array(nExperts).fill(0).map(() => ({
        w1: Array(nEmbd).fill(0).map(() =>
          Array(dFF).fill(0).map(() => (Math.random() - 0.5) * 0.3)
        ),
        w2: Array(dFF).fill(0).map(() =>
          Array(nEmbd).fill(0).map(() => (Math.random() - 0.5) * 0.3)
        ),
      })),
    };
  };

  const [inputData] = useState(generateInputData());
  const [weights] = useState(generateWeights());
  const [tokenTexts] = useState(['The', 'cat', 'sat']);

  const handleRun = () => {
    setIsRunning(true);
  };

  const handleReset = () => {
    setIsRunning(false);
    // 触发重新渲染
    setTimeout(() => setIsRunning(true), 100);
  };

  const handleComplete = () => {
    console.log('MoE FFN visualization completed');
  };

  return (
    <div className="w-full min-h-screen bg-gray-100 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-4">
            Mixture of Experts (MoE) Feed-Forward Network
          </h1>
          <p className="text-gray-600 mb-4">
            This visualization demonstrates how MoE FFN works in Transformer models. 
            Each token is routed to a subset of expert networks based on learned routing probabilities.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Number of Experts</div>
              <div className="text-2xl font-bold text-blue-600">{nExperts}</div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Top-K Selection</div>
              <div className="text-2xl font-bold text-green-600">{config.top_k}</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Tokens</div>
              <div className="text-2xl font-bold text-purple-600">{nTokens}</div>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Embedding Dim</div>
              <div className="text-2xl font-bold text-orange-600">{nEmbd}</div>
            </div>
          </div>

          <div className="flex gap-4">
            <button
              onClick={handleRun}
              disabled={isRunning}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {isRunning ? 'Running...' : 'Start Visualization'}
            </button>
            <button
              onClick={handleReset}
              disabled={!isRunning}
              className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              Restart
            </button>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold text-gray-800 mb-3">Key Concepts</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="border-l-4 border-blue-500 pl-4">
              <h3 className="font-bold text-gray-700 mb-1">Gating Network</h3>
              <p className="text-sm text-gray-600">
                Routes each token to the most suitable experts by computing selection probabilities
              </p>
            </div>
            <div className="border-l-4 border-green-500 pl-4">
              <h3 className="font-bold text-gray-700 mb-1">Expert Networks</h3>
              <p className="text-sm text-gray-600">
                Independent FFN modules that specialize in different aspects of the transformation
              </p>
            </div>
            <div className="border-l-4 border-purple-500 pl-4">
              <h3 className="font-bold text-gray-700 mb-1">Top-K Selection</h3>
              <p className="text-sm text-gray-600">
                Only the top-k experts with highest probabilities process each token (sparse activation)
              </p>
            </div>
            <div className="border-l-4 border-orange-500 pl-4">
              <h3 className="font-bold text-gray-700 mb-1">Load Balancing</h3>
              <p className="text-sm text-gray-600">
                Ensures all experts are utilized effectively to prevent some from being underused
              </p>
            </div>
          </div>
        </div>

        {isRunning && (
          <MoEFFNViz
            inputData={inputData}
            weights={weights}
            config={config}
            tokenTexts={tokenTexts}
            animationMode="serial"
            onComplete={handleComplete}
          />
        )}

        <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
          <h2 className="text-xl font-bold text-gray-800 mb-3">Advantages of MoE</h2>
          <ul className="list-disc list-inside space-y-2 text-gray-600">
            <li>
              <strong>Increased Model Capacity:</strong> More parameters without proportional increase in computation
            </li>
            <li>
              <strong>Conditional Computation:</strong> Different tokens can use different expert networks
            </li>
            <li>
              <strong>Specialization:</strong> Experts can specialize in different types of patterns or domains
            </li>
            <li>
              <strong>Efficiency:</strong> Sparse activation means only a subset of parameters are used per token
            </li>
            <li>
              <strong>Scalability:</strong> Easy to scale to very large models (e.g., GPT-4 with hundreds of experts)
            </li>
          </ul>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
          <h2 className="text-xl font-bold text-gray-800 mb-3">Visualization Steps</h2>
          <ol className="list-decimal list-inside space-y-2 text-gray-600">
            <li><strong>Layer Normalization:</strong> Normalize the input before routing</li>
            <li><strong>Gating Network:</strong> Compute expert selection logits</li>
            <li><strong>Softmax:</strong> Convert logits to probabilities</li>
            <li><strong>Expert Selection:</strong> Visualize selection probabilities with bar charts</li>
            <li><strong>Expert Networks:</strong> Display all available expert modules</li>
            <li><strong>Routing Animation:</strong> Show tokens being routed to selected experts</li>
            <li><strong>Expert Computation:</strong> Process tokens through expert FFN layers</li>
            <li><strong>Output Merging:</strong> Combine weighted expert outputs</li>
            <li><strong>Residual Connection:</strong> Add original input to final output</li>
            <li><strong>Load Balancing:</strong> Show expert usage statistics</li>
          </ol>
        </div>
      </div>
    </div>
  );
};
