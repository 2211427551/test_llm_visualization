import type { MoERoutingData, SparseAttentionCell } from '../types/visualization'

const average = (values: number[]) => {
  if (values.length === 0) {
    return 0
  }
  const total = values.reduce((sum, value) => sum + value, 0)
  return total / values.length
}

export const downsampleMatrix = (matrix: number[][], maxDimension = 48): number[][] => {
  if (matrix.length === 0 || matrix[0]?.length === 0) {
    return matrix
  }

  const rows = matrix.length
  const cols = matrix[0].length

  if (rows <= maxDimension && cols <= maxDimension) {
    return matrix
  }

  const rowGroupSize = Math.ceil(rows / maxDimension)
  const colGroupSize = Math.ceil(cols / maxDimension)

  const downsampled: number[][] = []
  for (let rowStart = 0; rowStart < rows; rowStart += rowGroupSize) {
    const rowSlice = matrix.slice(rowStart, rowStart + rowGroupSize)
    const aggregatedRow: number[] = []
    for (let colStart = 0; colStart < cols; colStart += colGroupSize) {
      const blockValues: number[] = []
      for (const row of rowSlice) {
        blockValues.push(...row.slice(colStart, colStart + colGroupSize))
      }
      aggregatedRow.push(Number(average(blockValues).toFixed(3)))
    }
    downsampled.push(aggregatedRow)
  }

  return downsampled
}

export const downsampleSparseMatrix = (
  matrix: SparseAttentionCell[][],
  maxDimension = 48,
): SparseAttentionCell[][] => {
  if (matrix.length === 0 || matrix[0]?.length === 0) {
    return matrix
  }

  if (matrix.length <= maxDimension && matrix[0].length <= maxDimension) {
    return matrix
  }

  const rows = matrix.length
  const cols = matrix[0].length
  const rowGroupSize = Math.ceil(rows / maxDimension)
  const colGroupSize = Math.ceil(cols / maxDimension)

  const downsampled: SparseAttentionCell[][] = []
  for (let rowStart = 0; rowStart < rows; rowStart += rowGroupSize) {
    const rowSlice = matrix.slice(rowStart, rowStart + rowGroupSize)
    const aggregatedRow: SparseAttentionCell[] = []
    for (let colStart = 0; colStart < cols; colStart += colGroupSize) {
      const block: SparseAttentionCell[] = []
      for (const row of rowSlice) {
        block.push(...row.slice(colStart, colStart + colGroupSize))
      }
      const values = block.map((cell) => cell.value)
      const sparseCount = block.filter((cell) => cell.isSparse).length
      const totalCount = block.length || 1
      aggregatedRow.push({
        value: Number(average(values).toFixed(3)),
        isSparse: sparseCount / totalCount > 0.5,
      })
    }
    downsampled.push(aggregatedRow)
  }

  return downsampled
}

export const limitMoERouting = (
  data: MoERoutingData,
  options: { maxTokens?: number; maxExperts?: number; maxRoutes?: number } = {},
): MoERoutingData => {
  const { maxTokens = 18, maxExperts = 18, maxRoutes = 160 } = options
  const tokens = data.tokens.slice(0, maxTokens)
  const tokenIds = new Set(tokens.map((token) => token.id))

  const filteredRoutes = data.routes
    .filter((route) => tokenIds.has(route.tokenId))
    .sort((a, b) => b.weight - a.weight)
    .slice(0, maxRoutes)

  const expertIds = new Set(filteredRoutes.map((route) => route.expertId))
  const experts = data.experts.filter((expert) => expertIds.has(expert.id)).slice(0, maxExperts)

  return {
    tokens,
    experts,
    routes: filteredRoutes.filter((route) => expertIds.has(route.expertId)),
    description: data.description,
  }
}
