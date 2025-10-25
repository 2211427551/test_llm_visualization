import { ModelForwardResponse, NormalizedModel, ModelNode, LayerType } from '../types';

function estimateParamCount(layerData: { inputShape?: number[], outputShape?: number[], layerType?: LayerType }): number {
  if (!layerData.inputShape || !layerData.outputShape) return 0;
  
  const inputDim = layerData.inputShape.reduce((a, b) => a * b, 1);
  const outputDim = layerData.outputShape.reduce((a, b) => a * b, 1);
  
  switch (layerData.layerType) {
    case 'embedding':
      return outputDim * 50000; // Vocab size estimate
    case 'attention':
      return inputDim * outputDim * 4; // Q, K, V, O projections
    case 'moe':
      return inputDim * outputDim * 8; // 8 experts estimate
    case 'feedforward':
      return inputDim * outputDim * 2;
    default:
      return inputDim * outputDim;
  }
}

export function normalizeModelResponse(response: ModelForwardResponse): NormalizedModel {
  const steps = response.steps;
  
  // Find embedding layer (usually first)
  const embeddingStep = steps.find(s => s.layerData.layerType === 'embedding');
  
  // Group encoder blocks (attention + feedforward/moe pairs)
  const encoderBlocks: ModelNode[] = [];
  let currentBlock: ModelNode | null = null;
  
  let totalParams = 0;
  let maxDimension = 0;
  
  steps.forEach((step) => {
    const layer = step.layerData;
    const paramCount = estimateParamCount(layer);
    totalParams += paramCount;
    
    if (layer.outputShape && layer.outputShape.length > 0) {
      maxDimension = Math.max(maxDimension, ...layer.outputShape);
    }
    
    if (layer.layerType === 'attention') {
      // Start new encoder block
      currentBlock = {
        id: `encoder-block-${encoderBlocks.length}`,
        type: 'attention',
        name: `Encoder Block ${encoderBlocks.length + 1}`,
        children: [{
          id: `layer-${layer.layerId}`,
          type: layer.layerType,
          name: layer.layerName,
          stepIndex: step.stepIndex,
          inputShape: layer.inputShape,
          outputShape: layer.outputShape,
          paramCount,
        }],
        inputShape: layer.inputShape,
        outputShape: layer.outputShape,
        paramCount,
      };
    } else if ((layer.layerType === 'moe' || layer.layerType === 'feedforward') && currentBlock) {
      // Add to current encoder block
      currentBlock.children!.push({
        id: `layer-${layer.layerId}`,
        type: layer.layerType,
        name: layer.layerName,
        stepIndex: step.stepIndex,
        inputShape: layer.inputShape,
        outputShape: layer.outputShape,
        paramCount,
      });
      currentBlock.outputShape = layer.outputShape;
      currentBlock.paramCount! += paramCount;
      encoderBlocks.push(currentBlock);
      currentBlock = null;
    }
  });
  
  // Handle orphaned attention blocks (no feedforward/moe after)
  if (currentBlock) {
    encoderBlocks.push(currentBlock);
  }
  
  // Find output layer (usually last)
  const outputStep = steps[steps.length - 1];
  
  const embedding: ModelNode = embeddingStep ? {
    id: `layer-${embeddingStep.layerData.layerId}`,
    type: 'embedding',
    name: embeddingStep.layerData.layerName,
    stepIndex: embeddingStep.stepIndex,
    inputShape: embeddingStep.layerData.inputShape,
    outputShape: embeddingStep.layerData.outputShape,
    paramCount: estimateParamCount(embeddingStep.layerData),
  } : {
    id: 'embedding-default',
    type: 'embedding',
    name: 'Embedding',
  };
  
  const output: ModelNode = {
    id: outputStep ? `layer-${outputStep.layerData.layerId}` : 'output-default',
    type: outputStep?.layerData.layerType || 'output',
    name: outputStep?.layerData.layerName || 'Output',
    stepIndex: outputStep?.stepIndex,
    inputShape: outputStep?.layerData.inputShape,
    outputShape: outputStep?.layerData.outputShape,
    paramCount: outputStep ? estimateParamCount(outputStep.layerData) : 0,
  };
  
  return {
    embedding,
    encoderBlocks,
    output,
    totalParams,
    maxDimension,
  };
}

export function getNodeColor(type: LayerType): string {
  switch (type) {
    case 'embedding':
      return '#3b82f6'; // blue
    case 'attention':
      return '#8b5cf6'; // purple
    case 'moe':
      return '#ec4899'; // pink
    case 'feedforward':
      return '#10b981'; // green
    case 'normalization':
      return '#f59e0b'; // amber
    case 'output':
      return '#ef4444'; // red
    default:
      return '#6b7280'; // gray
  }
}

export function formatParamCount(count: number): string {
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}B`;
  if (count >= 1e6) return `${(count / 1e6).toFixed(2)}M`;
  if (count >= 1e3) return `${(count / 1e3).toFixed(2)}K`;
  return count.toString();
}
