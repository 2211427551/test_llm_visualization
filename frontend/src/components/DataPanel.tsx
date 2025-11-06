import { useId, useState } from 'react'

interface DataPanelProps {
  title: string
  children: React.ReactNode
  defaultExpanded?: boolean
  scrollable?: boolean
}

const DataPanel = ({ title, children, defaultExpanded = true, scrollable = false }: DataPanelProps) => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const panelId = useId()
  const contentId = `${panelId}-content`

  return (
    <div className="rounded-lg border border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800">
      <button
        type="button"
        className="flex w-full items-center justify-between px-3 py-3 text-left hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 dark:hover:bg-slate-700"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        aria-controls={contentId}
      >
        <span className="text-sm font-medium text-slate-900 dark:text-slate-50">{title}</span>
        <span className="text-slate-400" aria-hidden="true">
          <svg
            className={`h-4 w-4 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </span>
      </button>

      {isExpanded && (
        <div
          id={contentId}
          role="region"
          aria-label={title}
          className={`border-t border-slate-200 dark:border-slate-700 ${scrollable ? 'max-h-64 overflow-y-auto' : ''}`}
        >
          <div className="p-3">{children}</div>
        </div>
      )}
    </div>
  )
}

export default DataPanel
