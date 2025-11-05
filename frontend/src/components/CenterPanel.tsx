import { useState } from 'react'

interface LayerNode {
  id: string
  name: string
  type: 'input' | 'embedding' | 'attention' | 'feedforward' | 'output'
  position: { x: number; y: number }
}

const CenterPanel = () => {
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null)

  // Static placeholder data for model structure
  const layers: LayerNode[] = [
    { id: '1', name: 'Input Layer', type: 'input', position: { x: 50, y: 50 } },
    { id: '2', name: 'Embedding', type: 'embedding', position: { x: 50, y: 150 } },
    { id: '3', name: 'Multi-Head Attention', type: 'attention', position: { x: 50, y: 250 } },
    { id: '4', name: 'Feed Forward', type: 'feedforward', position: { x: 50, y: 350 } },
    { id: '5', name: 'Output Layer', type: 'output', position: { x: 50, y: 450 } },
  ]

  const handleLayerClick = (layerId: string) => {
    setSelectedLayer(layerId)
    // This will trigger right panel update in the parent component
    console.log('Selected layer:', layerId)
  }

  const getLayerColor = (type: LayerNode['type']) => {
    switch (type) {
      case 'input':
        return 'bg-blue-100 border-blue-300 dark:bg-blue-900 dark:border-blue-700'
      case 'embedding':
        return 'bg-green-100 border-green-300 dark:bg-green-900 dark:border-green-700'
      case 'attention':
        return 'bg-purple-100 border-purple-300 dark:bg-purple-900 dark:border-purple-700'
      case 'feedforward':
        return 'bg-orange-100 border-orange-300 dark:bg-orange-900 dark:border-orange-700'
      case 'output':
        return 'bg-red-100 border-red-300 dark:bg-red-900 dark:border-red-700'
      default:
        return 'bg-gray-100 border-gray-300 dark:bg-gray-900 dark:border-gray-700'
    }
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">模型结构视图</h2>
        <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">点击各层查看详细信息</p>
      </div>

      {/* Model Structure Diagram */}
      <div className="flex-1 rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-900">
        <div className="relative h-full">
          {/* Placeholder for D3 integration */}
          <div className="flex h-full flex-col items-center justify-center">
            <div className="w-full max-w-md space-y-4">
              {layers.map((layer) => (
                <div
                  key={layer.id}
                  onClick={() => handleLayerClick(layer.id)}
                  className={`cursor-pointer rounded-lg border-2 p-4 text-center transition-all hover:shadow-lg ${getLayerColor(
                    layer.type,
                  )} ${
                    selectedLayer === layer.id
                      ? 'ring-2 ring-primary ring-offset-2 dark:ring-offset-slate-900'
                      : ''
                  }`}
                >
                  <div className="text-sm font-medium text-slate-900 dark:text-slate-50">
                    {layer.name}
                  </div>
                  <div className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                    Type: {layer.type}
                  </div>
                </div>
              ))}

              {/* Connection lines (visual placeholder) */}
              <div className="relative">
                <div className="absolute left-1/2 h-full w-0.5 -translate-x-1/2 bg-slate-300 dark:bg-slate-600" />
              </div>
            </div>
          </div>

          {/* Legend */}
          <div className="absolute bottom-4 right-4 rounded-lg bg-white p-3 shadow-lg dark:bg-slate-800">
            <div className="mb-2 text-xs font-medium text-slate-700 dark:text-slate-300">图例</div>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded border border-blue-300 bg-blue-100 dark:border-blue-700 dark:bg-blue-900" />
                <span className="text-xs text-slate-600 dark:text-slate-400">输入层</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded border border-green-300 bg-green-100 dark:border-green-700 dark:bg-green-900" />
                <span className="text-xs text-slate-600 dark:text-slate-400">嵌入层</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded border border-purple-300 bg-purple-100 dark:border-purple-700 dark:bg-purple-900" />
                <span className="text-xs text-slate-600 dark:text-slate-400">注意力层</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded border border-orange-300 bg-orange-100 dark:border-orange-700 dark:bg-orange-900" />
                <span className="text-xs text-slate-600 dark:text-slate-400">前馈层</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded border border-red-300 bg-red-100 dark:border-red-700 dark:bg-red-900" />
                <span className="text-xs text-slate-600 dark:text-slate-400">输出层</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Info Panel */}
      <div className="mt-4 rounded-lg bg-blue-50 p-3 dark:bg-blue-900/20">
        <div className="text-sm text-blue-800 dark:text-blue-200">
          <strong>提示：</strong>这是模型结构的静态示意图。后续将集成 D3.js 实现交互式可视化。
        </div>
      </div>
    </div>
  )
}

export default CenterPanel
