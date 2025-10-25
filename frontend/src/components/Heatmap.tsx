import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  calculateMatrixStats,
  getColorForValue,
  shouldDownsample,
  downsampleMatrix,
  formatNumber,
} from '../utils/heatmapUtils';

interface HeatmapProps {
  data: number[][];
  mask?: number[][];
  title?: string;
  rowLabels?: string[];
  colLabels?: string[];
  maxDisplaySize?: number;
}

export const Heatmap: React.FC<HeatmapProps> = ({
  data,
  mask,
  title,
  rowLabels,
  colLabels,
  maxDisplaySize = 16,
}) => {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  if (!data || data.length === 0) {
    return null;
  }

  const stats = calculateMatrixStats(data, mask);
  const needsDownsampling = shouldDownsample(data.length, data[0].length, maxDisplaySize * maxDisplaySize);

  let displayData = data;
  let displayMask = mask;
  let downsampledInfo = null;

  if (needsDownsampling) {
    const downsampled = downsampleMatrix(data, maxDisplaySize, maxDisplaySize);
    displayData = downsampled.matrix;
    downsampledInfo = {
      originalRows: data.length,
      originalCols: data[0].length,
      displayRows: downsampled.downsampledRows,
      displayCols: downsampled.downsampledCols,
    };
    // Downsample mask as well if present
    if (mask) {
      const downsampledMask = downsampleMatrix(mask, maxDisplaySize, maxDisplaySize);
      displayMask = downsampledMask.matrix.map(row => row.map(v => (v > 0.5 ? 1 : 0)));
    }
  }

  const rows = displayData.length;
  const cols = displayData[0]?.length || 0;
  const cellSize = Math.min(40, Math.max(20, 600 / Math.max(rows, cols)));

  return (
    <div className="space-y-3">
      {title && <h6 className="text-sm font-semibold text-gray-700">{title}</h6>}

      <div className="bg-gray-50 p-4 rounded-lg">
        <div className="flex gap-4 mb-3 text-xs text-gray-600">
          <div>
            <span className="font-semibold">Min:</span> {formatNumber(stats.min, 3)}
          </div>
          <div>
            <span className="font-semibold">Max:</span> {formatNumber(stats.max, 3)}
          </div>
          <div>
            <span className="font-semibold">Mean:</span> {formatNumber(stats.mean, 3)}
          </div>
          <div>
            <span className="font-semibold">Std:</span> {formatNumber(stats.std, 3)}
          </div>
          {stats.sparsity > 0 && (
            <div>
              <span className="font-semibold">Sparsity:</span> {(stats.sparsity * 100).toFixed(1)}%
            </div>
          )}
        </div>

        {downsampledInfo && (
          <div className="mb-3 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2 py-1">
            ⚠️ Downsampled from {downsampledInfo.originalRows}×{downsampledInfo.originalCols} to{' '}
            {downsampledInfo.displayRows}×{downsampledInfo.displayCols} for performance
          </div>
        )}

        <div className="overflow-auto">
          <div
            className="inline-grid gap-[1px] bg-gray-300"
            style={{
              gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
            }}
          >
            {displayData.map((row, rowIdx) =>
              row.map((value, colIdx) => {
                const isMasked = displayMask && displayMask[rowIdx] && displayMask[rowIdx][colIdx] === 0;
                const isHovered = hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx;
                const bgColor = getColorForValue(value, stats.min, stats.max, isMasked);

                return (
                  <motion.div
                    key={`${rowIdx}-${colIdx}`}
                    className="relative flex items-center justify-center cursor-pointer"
                    style={{
                      width: `${cellSize}px`,
                      height: `${cellSize}px`,
                      backgroundColor: bgColor,
                    }}
                    onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                    onMouseLeave={() => setHoveredCell(null)}
                    whileHover={{ scale: 1.1, zIndex: 10 }}
                    transition={{ duration: 0.1 }}
                  >
                    {cellSize >= 30 && (
                      <span
                        className="text-[9px] font-mono"
                        style={{
                          color: isMasked ? '#9ca3af' : Math.abs(value) > (stats.max - stats.min) * 0.7 ? '#fff' : '#111',
                        }}
                      >
                        {formatNumber(value, 2)}
                      </span>
                    )}
                    {isHovered && (
                      <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap z-20 shadow-lg"
                      >
                        <div>
                          [{rowLabels?.[rowIdx] || rowIdx}, {colLabels?.[colIdx] || colIdx}]
                        </div>
                        <div className="font-semibold">{formatNumber(value, 6)}</div>
                        {isMasked && <div className="text-red-400">Masked (pruned)</div>}
                      </motion.div>
                    )}
                  </motion.div>
                );
              })
            )}
          </div>
        </div>

        <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
          <div>
            {rows} × {cols} matrix
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3" style={{ backgroundColor: getColorForValue(stats.min, stats.min, stats.max) }} />
              <span>Low</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3" style={{ backgroundColor: getColorForValue(stats.max, stats.min, stats.max) }} />
              <span>High</span>
            </div>
            {stats.sparsity > 0 && (
              <div className="flex items-center gap-1">
                <div className="w-3 h-3 bg-gray-100 border border-gray-300" />
                <span>Masked</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
