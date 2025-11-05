import { useState } from 'react'

interface DataPanelProps {
  title: string
  children: React.ReactNode
  defaultExpanded?: boolean
  scrollable?: boolean
}

const DataPanel = ({
  title,
  children,
  defaultExpanded = true,
  scrollable = false,
}: DataPanelProps) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)

  return (
    <div className="rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
      {/* Header */}
      <div
        className="flex cursor-pointer items-center justify-between p-3 hover:bg-slate-50 dark:hover:bg-slate-700"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h3 className="text-sm font-medium text-slate-900 dark:text-slate-50">{title}</h3>
        <button className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300">
          <svg
            className={`h-4 w-4 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
      </div>

      {/* Content */}
      {isExpanded && (
        <div
          className={`border-t border-slate-200 dark:border-slate-700 ${
            scrollable ? 'max-h-64 overflow-y-auto' : ''
          }`}
        >
          <div className="p-3">{children}</div>
        </div>
      )}
    </div>
  )
}

export default DataPanel
