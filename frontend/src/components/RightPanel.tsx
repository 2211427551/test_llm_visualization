import { useMemo } from 'react'
import DataPanel from './DataPanel'
import TensorHeatmap from './visualizations/TensorHeatmap'
import SparseAttentionMatrix from './visualizations/SparseAttentionMatrix'
import MoERoutingDiagram from './visualizations/MoERoutingDiagram'
import { useVisualizationState } from '../hooks/useVisualizationState'
import type { LayerType } from '../types/visualization'

const layerTypeLabelMap: Record<LayerType, string> = {
  input: '输入层',
  embedding: '嵌入层',
  attention: '注意力层',
  feedforward: '前馈层',
  output: '输出层',
}

const RightPanel = () => {
  const { currentStep, currentStepIndex, stepCount, selectedLayer, runtimeSummary, tokenSequence } =
    useVisualizationState()

  const runtimeMetrics = useMemo(
    () =>
      runtimeSummary
        ? [
            { label: '前向传播时间', value: `${runtimeSummary.forwardTimeMs.toFixed(2)} ms` },
            { label: '内存使用', value: `${runtimeSummary.memoryMB.toFixed(1)} MB` },
            {
              label: 'GPU 利用率',
              value: `${Math.round(runtimeSummary.gpuUtilization * 100)}%`,
            },
            { label: '批处理大小', value: runtimeSummary.batchSize },
            { label: '序列长度', value: runtimeSummary.sequenceLength },
            { label: 'Logits 形状', value: runtimeSummary.logitsShape.join(' × ') },
          ]
        : [
            { label: '前向传播时间', value: '--' },
            { label: '内存使用', value: '--' },
            { label: 'GPU 利用率', value: '--' },
            { label: '批处理大小', value: '--' },
            { label: '序列长度', value: '--' },
            { label: 'Logits 形状', value: '--' },
          ],
    [runtimeSummary],
  )

  const displayedTokens = useMemo(() => tokenSequence.slice(0, 24), [tokenSequence])

  if (!selectedLayer) {
    return (
      <div className="flex h-full flex-col gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">细节展示</h2>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
            请选择任意 Transformer 层以查看对应的张量、注意力与路由信息。
          </p>
        </div>
        <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-300">
          暂无数据。
        </div>
      </div>
    )
  }

  const layerTypeLabel = layerTypeLabelMap[selectedLayer.type] ?? selectedLayer.type

  return (
    <div className="flex h-full flex-col gap-4">
      <div>
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">细节展示</h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
          动态渲染张量热力图、稀疏注意力与 MoE 路由，支持步骤逐帧切换。
        </p>
      </div>

      <div
        className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900"
        role="status"
        aria-live="polite"
      >
        <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
          <div>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              步骤 {currentStepIndex + 1}/{stepCount} · {currentStep.name}
            </p>
            <p className="text-base font-semibold text-slate-900 dark:text-slate-50">当前层：{selectedLayer.name}</p>
          </div>
          <span className="text-xs text-slate-500 dark:text-slate-400">{layerTypeLabel}</span>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-slate-600 dark:text-slate-300">{selectedLayer.summary}</p>
        {selectedLayer.sparseAttention?.note && (
          <p className="mt-2 rounded-md bg-slate-100 px-3 py-2 text-xs text-slate-600 dark:bg-slate-800/60 dark:text-slate-300">
            {selectedLayer.sparseAttention.note}
          </p>
        )}
        {selectedLayer.moeRouting?.description && (
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">{selectedLayer.moeRouting.description}</p>
        )}
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto" role="region" aria-label="层级数据详情">
        <DataPanel title="张量热力图" scrollable={false}>
          <div className="space-y-2">
            <TensorHeatmap data={selectedLayer.tensorHeatmap} ariaLabel={`张量热力图 · ${selectedLayer.name}`} />
            <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
              <span>0.0</span>
              <span>权重热度</span>
              <span>1.0</span>
            </div>
          </div>
        </DataPanel>

        <DataPanel title="稀疏注意力矩阵" scrollable={false}>
          {selectedLayer.sparseAttention ? (
            <div className="space-y-3">
              <SparseAttentionMatrix
                data={selectedLayer.sparseAttention}
                ariaLabel={`稀疏注意力矩阵 · ${selectedLayer.name}`}
              />
              <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
                <span className="flex items-center gap-1">
                  <span
                    className="inline-block h-3 w-3 rounded-sm border border-slate-300"
                    style={{ backgroundColor: '#e2e8f0' }}
                    aria-hidden="true"
                  />
                  稀疏区域
                </span>
                <span className="flex items-center gap-1">
                  <span
                    className="inline-block h-3 w-3 rounded-sm border border-purple-400"
                    style={{ backgroundColor: '#c084fc' }}
                    aria-hidden="true"
                  />
                  活跃连接
                </span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500 dark:text-slate-400">本层不包含稀疏注意力权重。</p>
          )}
        </DataPanel>

        <DataPanel title="MoE 路由信息" scrollable={false}>
          {selectedLayer.moeRouting ? (
            <div className="space-y-3">
              <MoERoutingDiagram
                data={selectedLayer.moeRouting}
                ariaLabel={`MoE 路由流向图 · ${selectedLayer.name}`}
              />
              <div className="rounded-md bg-slate-100 px-3 py-2 text-xs text-slate-600 dark:bg-slate-800/60 dark:text-slate-300">
                <p>令牌数：{selectedLayer.moeRouting.tokens.length}</p>
                <p>专家数：{selectedLayer.moeRouting.experts.length}</p>
                <p>总路由：{selectedLayer.moeRouting.routes.length}</p>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500 dark:text-slate-400">本层未启用混合专家路由。</p>
          )}
        </DataPanel>

        <DataPanel title="运行统计" defaultExpanded={Boolean(runtimeSummary)}>
          <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
            {runtimeMetrics.map((metric) => (
              <div key={metric.label} className="flex justify-between">
                <span>{metric.label}:</span>
                <span>{metric.value}</span>
              </div>
            ))}
          </div>

          {tokenSequence.length > 0 && (
            <div className="mt-3 space-y-2 text-xs text-slate-500 dark:text-slate-400">
              <div className="font-medium text-slate-600 dark:text-slate-300">输入 Token</div>
              <div className="flex flex-wrap gap-2">
                {displayedTokens.map((token, index) => (
                  <span
                    key={`${token}-${index}`}
                    className="rounded-full bg-slate-100 px-2 py-1 text-slate-600 dark:bg-slate-800 dark:text-slate-200"
                  >
                    {token}
                  </span>
                ))}
                {tokenSequence.length > displayedTokens.length && (
                  <span className="text-slate-400 dark:text-slate-500">
                    +{tokenSequence.length - displayedTokens.length}
                  </span>
                )}
              </div>
            </div>
          )}

          {!runtimeSummary && tokenSequence.length === 0 && (
            <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">尚未执行推理。</p>
          )}
        </DataPanel>
      </div>
    </div>
  )
}

export default RightPanel
