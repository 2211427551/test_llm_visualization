export interface HeatmapCell {
  row: number;
  col: number;
  value: number;
  masked?: boolean;
}

export interface HeatmapStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  sparsity: number;
}

export function calculateMatrixStats(matrix: number[][], mask?: number[][]): HeatmapStats {
  const values: number[] = [];
  let maskedCount = 0;
  let totalCount = 0;

  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      totalCount++;
      const isMasked = mask && mask[i] && mask[i][j] === 0;
      if (!isMasked) {
        values.push(matrix[i][j]);
      } else {
        maskedCount++;
      }
    }
  }

  if (values.length === 0) {
    return { min: 0, max: 0, mean: 0, std: 0, sparsity: 1 };
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
  const std = Math.sqrt(variance);
  const sparsity = maskedCount / totalCount;

  return { min, max, mean, std, sparsity };
}

export function getColorForValue(
  value: number,
  min: number,
  max: number,
  isMasked: boolean = false
): string {
  if (isMasked) {
    return '#f3f4f6'; // gray-100
  }

  const range = max - min;
  if (range === 0) {
    return '#93c5fd'; // blue-300
  }

  const normalized = (value - min) / range;
  
  // Blue to red color scale
  if (normalized < 0.5) {
    // Blue to white
    const t = normalized * 2;
    const r = Math.round(147 + (255 - 147) * t);
    const g = Math.round(197 + (255 - 197) * t);
    const b = Math.round(253 + (255 - 253) * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    // White to red
    const t = (normalized - 0.5) * 2;
    const r = 255;
    const g = Math.round(255 - 108 * t);
    const b = Math.round(255 - 171 * t);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

export function shouldDownsample(rows: number, cols: number, maxCells: number = 256): boolean {
  return rows * cols > maxCells;
}

export function downsampleMatrix(
  matrix: number[][],
  maxRows: number = 16,
  maxCols: number = 16
): { matrix: number[][]; downsampledRows: number; downsampledCols: number } {
  const rows = matrix.length;
  const cols = matrix[0]?.length || 0;

  if (rows <= maxRows && cols <= maxCols) {
    return { matrix, downsampledRows: rows, downsampledCols: cols };
  }

  const rowStep = Math.ceil(rows / maxRows);
  const colStep = Math.ceil(cols / maxCols);

  const downsampled: number[][] = [];
  for (let i = 0; i < rows; i += rowStep) {
    const row: number[] = [];
    for (let j = 0; j < cols; j += colStep) {
      // Average the values in the block
      let sum = 0;
      let count = 0;
      for (let di = 0; di < rowStep && i + di < rows; di++) {
        for (let dj = 0; dj < colStep && j + dj < cols; dj++) {
          sum += matrix[i + di][j + dj];
          count++;
        }
      }
      row.push(sum / count);
    }
    downsampled.push(row);
  }

  return {
    matrix: downsampled,
    downsampledRows: downsampled.length,
    downsampledCols: downsampled[0]?.length || 0,
  };
}

export function formatNumber(value: number, precision: number = 4): string {
  if (Math.abs(value) < 0.0001 && value !== 0) {
    return value.toExponential(2);
  }
  return value.toFixed(precision);
}
