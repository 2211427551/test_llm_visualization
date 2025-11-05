import type { ReactNode } from 'react'

interface ThreeColumnLayoutProps {
  leftPanel: ReactNode
  centerPanel: ReactNode
  rightPanel: ReactNode
}

const ThreeColumnLayout = ({ leftPanel, centerPanel, rightPanel }: ThreeColumnLayoutProps) => {
  return (
    <div className="flex h-full gap-4">
      {/* Left Panel - Input & Controls */}
      <div className="w-80 min-w-80 flex-shrink-0">
        <div className="h-full rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
          {leftPanel}
        </div>
      </div>

      {/* Center Panel - Model Structure View */}
      <div className="min-w-0 flex-1">
        <div className="h-full rounded-lg border border-slate-200 bg-white p-6 dark:border-slate-700 dark:bg-slate-800">
          {centerPanel}
        </div>
      </div>

      {/* Right Panel - Details Display */}
      <div className="w-96 min-w-80 max-w-96 flex-shrink-0">
        <div className="h-full rounded-lg border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800">
          {rightPanel}
        </div>
      </div>
    </div>
  )
}

export default ThreeColumnLayout
