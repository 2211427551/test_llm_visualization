import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useExecutionStore } from '../store/executionStore';
import { normalizeModelResponse, formatParamCount } from '../utils/modelNormalizer';
import { ModelNode, LayerType } from '../types';
import { Info } from 'lucide-react';

interface NodeDimensions {
  x: number;
  y: number;
  width: number;
  height: number;
  node: ModelNode;
}

const LAYER_COLORS: Record<LayerType, { bg: string; border: string; text: string }> = {
  embedding: { bg: '#dbeafe', border: '#3b82f6', text: '#1e40af' },
  attention: { bg: '#ede9fe', border: '#8b5cf6', text: '#5b21b6' },
  moe: { bg: '#fce7f3', border: '#ec4899', text: '#9f1239' },
  feedforward: { bg: '#d1fae5', border: '#10b981', text: '#065f46' },
  normalization: { bg: '#fef3c7', border: '#f59e0b', text: '#92400e' },
  output: { bg: '#fee2e2', border: '#ef4444', text: '#991b1b' },
};

export const ModelOverview: React.FC = () => {
  const { data, currentStepIndex, selectedLayerId, setSelectedLayer, setCurrentStep, setBreadcrumbs } = useExecutionStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [focusedNode, setFocusedNode] = useState<string | null>(null);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({
          width: rect.width || 800,
          height: rect.height || 600,
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  if (!data) {
    return (
      <div 
        ref={containerRef}
        className="flex items-center justify-center h-full min-h-96 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300"
      >
        <div className="text-center">
          <Info className="mx-auto mb-2 text-gray-400" size={32} />
          <p className="text-gray-500">Run a model to see the architecture overview</p>
        </div>
      </div>
    );
  }

  const normalizedModel = normalizeModelResponse(data);
  const { embedding, encoderBlocks, output, totalParams, maxDimension } = normalizedModel;

  // Calculate layout
  const padding = 40;
  const nodeWidth = 140;
  const nodeHeight = 80;
  const blockHeight = 100;
  const verticalSpacing = 60;
  const horizontalSpacing = 20;
  
  const totalHeight = padding * 2 + nodeHeight + verticalSpacing + 
                     (encoderBlocks.length * (blockHeight + verticalSpacing)) + 
                     verticalSpacing + nodeHeight;

  const centerX = dimensions.width / 2;
  let currentY = padding;

  const nodeDimensions: NodeDimensions[] = [];

  // Embedding at bottom (visually, but we'll render top-to-bottom)
  nodeDimensions.push({
    x: centerX - nodeWidth / 2,
    y: currentY,
    width: nodeWidth,
    height: nodeHeight,
    node: embedding,
  });
  currentY += nodeHeight + verticalSpacing;

  // Encoder blocks
  encoderBlocks.forEach((block) => {
    const blockY = currentY;
    const childCount = block.children?.length || 0;
    const blockWidth = childCount * nodeWidth + (childCount - 1) * horizontalSpacing;
    const blockX = centerX - blockWidth / 2;

    if (block.children) {
      block.children.forEach((child, idx) => {
        nodeDimensions.push({
          x: blockX + idx * (nodeWidth + horizontalSpacing),
          y: blockY,
          width: nodeWidth,
          height: nodeHeight,
          node: child,
        });
      });
    }

    currentY += blockHeight + verticalSpacing;
  });

  // Output at top
  nodeDimensions.push({
    x: centerX - nodeWidth / 2,
    y: currentY,
    width: nodeWidth,
    height: nodeHeight,
    node: output,
  });

  const handleNodeClick = (node: ModelNode, event: React.MouseEvent | React.KeyboardEvent) => {
    event.preventDefault();
    setSelectedLayer(node.id);
    
    if (node.stepIndex !== undefined) {
      setCurrentStep(node.stepIndex);
    }

    // Build breadcrumb trail
    const breadcrumbs: string[] = [];
    if (node.type === 'embedding') {
      breadcrumbs.push('Embedding');
    } else if (encoderBlocks.some(block => block.children?.some(c => c.id === node.id))) {
      const blockIdx = encoderBlocks.findIndex(block => block.children?.some(c => c.id === node.id));
      breadcrumbs.push(`Encoder ${blockIdx + 1}`, node.name);
    } else {
      breadcrumbs.push(node.name);
    }
    setBreadcrumbs(breadcrumbs);
  };

  const handleKeyDown = (node: ModelNode, event: React.KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      handleNodeClick(node, event);
    }
  };

  const isNodeActive = (node: ModelNode): boolean => {
    return node.stepIndex !== undefined && node.stepIndex === currentStepIndex;
  };

  const isNodeSelected = (node: ModelNode): boolean => {
    return node.id === selectedLayerId;
  };

  return (
    <div ref={containerRef} className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm h-full overflow-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">Model Architecture</h3>
          <p className="text-sm text-gray-600">
            {encoderBlocks.length} Encoder Block{encoderBlocks.length !== 1 ? 's' : ''} • {formatParamCount(totalParams)} parameters • Max dimension: {maxDimension}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Legend />
        </div>
      </div>

      <svg
        width={dimensions.width}
        height={Math.max(totalHeight, dimensions.height - 120)}
        className="mx-auto"
        role="img"
        aria-label="Transformer model architecture diagram"
      >
        <defs>
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="10"
            refX="9"
            refY="3"
            orient="auto"
          >
            <polygon points="0 0, 10 3, 0 6" fill="#6b7280" />
          </marker>
        </defs>

        {/* Draw connections */}
        <g className="connections">
          {nodeDimensions.map((dim, idx) => {
            if (idx < nodeDimensions.length - 1) {
              const nextDim = nodeDimensions[idx + 1];
              const startX = dim.x + dim.width / 2;
              const startY = dim.y + dim.height;
              const endX = nextDim.x + nextDim.width / 2;
              const endY = nextDim.y;

              const isActiveConnection = isNodeActive(dim.node) || isNodeActive(nextDim.node);

              return (
                <motion.line
                  key={`connection-${dim.node.id}-${nextDim.node.id}`}
                  x1={startX}
                  y1={startY}
                  x2={endX}
                  y2={endY}
                  stroke={isActiveConnection ? '#3b82f6' : '#d1d5db'}
                  strokeWidth={isActiveConnection ? 3 : 2}
                  markerEnd="url(#arrowhead)"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.5, delay: idx * 0.1 }}
                />
              );
            }
            return null;
          })}
        </g>

        {/* Draw nodes */}
        <g className="nodes">
          <AnimatePresence>
            {nodeDimensions.map((dim, idx) => {
              const colors = LAYER_COLORS[dim.node.type];
              const isActive = isNodeActive(dim.node);
              const isSelected = isNodeSelected(dim.node);

              return (
                <g
                  key={dim.node.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${dim.node.name} layer, ${dim.node.type} type`}
                  aria-pressed={isSelected}
                  onKeyDown={(e) => handleKeyDown(dim.node, e)}
                  style={{ cursor: 'pointer' }}
                >
                  <motion.rect
                    x={dim.x}
                    y={dim.y}
                    width={dim.width}
                    height={dim.height}
                    rx={8}
                    fill={colors.bg}
                    stroke={isActive || isSelected ? colors.border : '#d1d5db'}
                    strokeWidth={isActive || isSelected ? 3 : 2}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ 
                      opacity: 1, 
                      scale: isActive ? 1.05 : 1,
                      y: dim.y,
                    }}
                    transition={{ duration: 0.3, delay: idx * 0.05 }}
                    whileHover={{ scale: 1.05 }}
                    onMouseEnter={() => setFocusedNode(dim.node.id)}
                    onMouseLeave={() => setFocusedNode(null)}
                    onClick={(e) => handleNodeClick(dim.node, e as unknown as React.MouseEvent)}
                  />
                  
                  {/* Node content */}
                  <motion.text
                    x={dim.x + dim.width / 2}
                    y={dim.y + 25}
                    textAnchor="middle"
                    className="font-semibold"
                    fill={colors.text}
                    fontSize={14}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: idx * 0.05 + 0.2 }}
                    pointerEvents="none"
                  >
                    {dim.node.name}
                  </motion.text>

                  {dim.node.outputShape && (
                    <motion.text
                      x={dim.x + dim.width / 2}
                      y={dim.y + 45}
                      textAnchor="middle"
                      fill="#6b7280"
                      fontSize={11}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.05 + 0.3 }}
                      pointerEvents="none"
                    >
                      {dim.node.outputShape.join(' × ')}
                    </motion.text>
                  )}

                  {dim.node.paramCount && (
                    <motion.text
                      x={dim.x + dim.width / 2}
                      y={dim.y + 60}
                      textAnchor="middle"
                      fill="#9ca3af"
                      fontSize={10}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: idx * 0.05 + 0.4 }}
                      pointerEvents="none"
                    >
                      {formatParamCount(dim.node.paramCount)}
                    </motion.text>
                  )}

                  {(isActive || isSelected) && (
                    <motion.rect
                      x={dim.x - 2}
                      y={dim.y - 2}
                      width={dim.width + 4}
                      height={dim.height + 4}
                      rx={10}
                      fill="none"
                      stroke={colors.border}
                      strokeWidth={2}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 0.6 }}
                      exit={{ opacity: 0 }}
                      pointerEvents="none"
                    />
                  )}
                </g>
              );
            })}
          </AnimatePresence>
        </g>
      </svg>

      {/* Tooltip for hovered node */}
      <AnimatePresence>
        {focusedNode && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="fixed bottom-4 right-4 bg-gray-900 text-white p-3 rounded-lg shadow-lg max-w-xs z-50"
          >
            {(() => {
              const node = nodeDimensions.find(d => d.node.id === focusedNode)?.node;
              if (!node) return null;
              return (
                <div className="text-sm">
                  <div className="font-semibold mb-1">{node.name}</div>
                  <div className="text-gray-300 text-xs space-y-1">
                    <div>Type: {node.type}</div>
                    {node.inputShape && <div>Input: {node.inputShape.join(' × ')}</div>}
                    {node.outputShape && <div>Output: {node.outputShape.join(' × ')}</div>}
                    {node.paramCount && <div>Parameters: {formatParamCount(node.paramCount)}</div>}
                  </div>
                </div>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const Legend: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1 text-sm text-gray-600 hover:text-gray-800 border border-gray-300 rounded-md hover:bg-gray-50"
        aria-label="Toggle color legend"
        aria-expanded={isOpen}
      >
        <Info size={16} />
        Legend
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            className="absolute right-0 top-full mt-2 bg-white border border-gray-200 rounded-lg shadow-lg p-4 z-50 min-w-48"
          >
            <h4 className="font-semibold text-sm mb-2 text-gray-800">Layer Types</h4>
            <div className="space-y-2">
              {Object.entries(LAYER_COLORS).map(([type, colors]) => (
                <div key={type} className="flex items-center gap-2">
                  <div
                    className="w-4 h-4 rounded border-2"
                    style={{ backgroundColor: colors.bg, borderColor: colors.border }}
                  />
                  <span className="text-sm text-gray-700 capitalize">{type}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
