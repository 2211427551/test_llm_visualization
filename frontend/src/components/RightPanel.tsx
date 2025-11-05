import DataPanel from './DataPanel'

const RightPanel = () => {
  // Placeholder data for tensor heatmap
  const renderTensorHeatmap = () => {
    const rows = 8
    const cols = 8
    const data = Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => Math.random()),
    )

    return (
      <div className="space-y-2">
        <div className="text-xs text-slate-600 dark:text-slate-400">张量维度: [8, 8]</div>
        <div className="grid grid-cols-8 gap-0.5">
          {data.map((row, i) =>
            row.map((value, j) => (
              <div
                key={`${i}-${j}`}
                className="aspect-square rounded-sm"
                style={{
                  backgroundColor: `rgba(59, 130, 246, ${value})`, // Blue gradient
                }}
                title={`[${i},${j}]: ${value.toFixed(3)}`}
              />
            )),
          )}
        </div>
        <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-400">
          <span>0.0</span>
          <span>张量热力图</span>
          <span>1.0</span>
        </div>
      </div>
    )
  }

  // Placeholder data for attention matrix
  const renderAttentionMatrix = () => {
    const size = 6
    const data = Array.from({ length: size }, (_, i) =>
      Array.from({ length: size }, (_, j) => {
        // Create some pattern in the attention matrix
        if (i === j) return 0.8 // Self-attention
        if (Math.abs(i - j) === 1) return 0.4 // Adjacent attention
        return Math.random() * 0.2
      }),
    )

    return (
      <div className="space-y-2">
        <div className="text-xs text-slate-600 dark:text-slate-400">注意力矩阵: [6, 6]</div>
        <div className="grid grid-cols-6 gap-0.5">
          {data.map((row, i) =>
            row.map((value, j) => (
              <div
                key={`${i}-${j}`}
                className="aspect-square rounded-sm"
                style={{
                  backgroundColor: `rgba(168, 85, 247, ${value})`, // Purple gradient
                }}
                title={`Attention[${i},${j}]: ${value.toFixed(3)}`}
              />
            )),
          )}
        </div>
        <div className="flex items-center justify-between text-xs text-slate-600 dark:text-slate-400">
          <span>0.0</span>
          <span>注意力权重</span>
          <span>1.0</span>
        </div>
      </div>
    )
  }

  // Placeholder data for MoE routing information
  const renderMoERouting = () => {
    const experts = ['专家 1', '专家 2', '专家 3', '专家 4']
    const routingWeights = [0.35, 0.25, 0.25, 0.15]

    return (
      <div className="space-y-3">
        <div className="text-xs text-slate-600 dark:text-slate-400">MoE 路由信息</div>
        <div className="space-y-2">
          {experts.map((expert, i) => (
            <div key={expert} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-slate-700 dark:text-slate-300">{expert}</span>
                <span className="text-slate-600 dark:text-slate-400">
                  {(routingWeights[i] * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 w-full rounded-full bg-slate-200 dark:bg-slate-700">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-primary to-secondary"
                  style={{ width: `${routingWeights[i] * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-3 rounded-lg bg-slate-50 p-2 dark:bg-slate-700">
          <div className="text-xs text-slate-600 dark:text-slate-400">
            <div>总负载: 100%</div>
            <div>激活专家数: 2/4</div>
            <div>负载均衡系数: 1.23</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">细节展示</h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">模型运行时的详细信息</p>
      </div>

      {/* Data Panels */}
      <div className="flex-1 space-y-4 overflow-y-auto">
        {/* Tensor Heatmap Panel */}
        <DataPanel title="张量热力图" scrollable={false}>
          {renderTensorHeatmap()}
        </DataPanel>

        {/* Attention Matrix Panel */}
        <DataPanel title="注意力矩阵" scrollable={false}>
          {renderAttentionMatrix()}
        </DataPanel>

        {/* MoE Routing Panel */}
        <DataPanel title="MoE 路由信息" scrollable={true}>
          {renderMoERouting()}
        </DataPanel>

        {/* Additional Info Panel */}
        <DataPanel title="运行统计" defaultExpanded={false}>
          <div className="space-y-2 text-sm text-slate-600 dark:text-slate-400">
            <div className="flex justify-between">
              <span>前向传播时间:</span>
              <span>12.3ms</span>
            </div>
            <div className="flex justify-between">
              <span>内存使用:</span>
              <span>2.1GB</span>
            </div>
            <div className="flex justify-between">
              <span>GPU 利用率:</span>
              <span>78%</span>
            </div>
            <div className="flex justify-between">
              <span>批处理大小:</span>
              <span>32</span>
            </div>
            <div className="flex justify-between">
              <span>序列长度:</span>
              <span>512</span>
            </div>
          </div>
        </DataPanel>
      </div>
    </div>
  )
}

export default RightPanel
