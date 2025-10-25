# Frontend Scaffold Setup Summary

This document outlines the React frontend foundation that has been set up for the Model Execution Visualizer.

## ✅ Completed Setup Tasks

### 1. Project Initialization
- ✅ Created `frontend/` directory with Vite + React + TypeScript
- ✅ Configured modern build tooling with Vite 5.x
- ✅ TypeScript strict mode enabled with proper compiler options

### 2. Code Quality Tools
- ✅ ESLint configured with React and TypeScript plugins
- ✅ Prettier configured with consistent code formatting rules
- ✅ Git hooks ready (via scripts in package.json)

### 3. Path Aliases
Absolute import aliases configured in both `vite.config.ts` and `tsconfig.json`:
```typescript
import { HomeView } from '@views/HomeView';
import { runModelForward } from '@api/client';
import { useExecutionStore } from '@store/executionStore';
```

Available aliases:
- `@/` - src root
- `@components/` - components directory
- `@views/` - views directory  
- `@api/` - API client directory
- `@store/` - state management
- `@hooks/` - custom hooks
- `@utils/` - utility functions

**Note:** Use relative imports for the `types` directory to avoid TypeScript conflicts.

### 4. Required Packages Installed

#### Core Dependencies:
- `react` & `react-dom` (^18.2.0)
- `d3` (^7.8.5) - for data visualization
- `zustand` (^4.4.7) - for state management
- `@tanstack/react-query` (^5.17.9) - for API state management
- `framer-motion` (^10.16.16) - for animations
- `axios` (^1.6.2) - for HTTP requests
- `lucide-react` (^0.298.0) - for icons

#### Styling:
- `tailwindcss` (^3.4.0)
- `postcss` & `autoprefixer`

#### Dev Dependencies:
- TypeScript & type definitions
- ESLint with React plugins
- Prettier
- Vite & React plugin
- `@types/d3` & `@types/node`

### 5. Folder Structure

```
frontend/
├── src/
│   ├── api/              # API client & react-query hooks
│   │   ├── client.ts     # Axios instance & API methods
│   │   └── queries.ts    # React Query hooks
│   ├── components/       # Reusable UI components
│   │   ├── TextInput.tsx
│   │   ├── ExecutionControls.tsx
│   │   ├── MacroView.tsx
│   │   ├── MicroView.tsx
│   │   ├── ModelOverview.tsx
│   │   ├── SummaryPanel.tsx
│   │   ├── ErrorDisplay.tsx
│   │   ├── Breadcrumb.tsx
│   │   ├── Heatmap.tsx
│   │   ├── AttentionInspector.tsx
│   │   └── MoEInspector.tsx
│   ├── views/            # Page-level view components
│   │   ├── HomeView.tsx  # Main application view
│   │   └── index.ts      # Barrel exports
│   ├── store/            # Zustand state management
│   │   └── executionStore.ts
│   ├── hooks/            # Custom React hooks
│   │   ├── useKeyboardShortcuts.ts
│   │   └── usePlayback.ts
│   ├── types/            # TypeScript type definitions
│   │   └── index.ts
│   ├── utils/            # Utility functions
│   │   └── d3-helpers.ts # D3 utility functions
│   ├── App.tsx           # Root component
│   ├── main.tsx          # Entry point with providers
│   └── index.css         # Global styles (Tailwind)
├── .env                  # Environment variables (gitignored)
├── .env.example          # Environment template
├── .eslintrc.cjs         # ESLint configuration
├── .prettierrc           # Prettier configuration
├── .gitignore            # Git ignore rules
├── tsconfig.json         # TypeScript config with path aliases
├── vite.config.ts        # Vite config with path aliases
├── tailwind.config.js    # Tailwind theme configuration
├── postcss.config.js     # PostCSS configuration
└── package.json          # Dependencies & scripts
```

### 6. Global Theme & Styling
- TailwindCSS configured with custom color palette
- Custom `primary` color shades defined in `tailwind.config.js`
- Global styles in `index.css` with Tailwind directives
- Responsive design utilities available

### 7. Layout Shell with Split Panes
The application features a responsive split-pane layout:
- **Left Pane (Macro/Micro Views)**: 2/3 width on large screens
  - Architecture view or List view toggle
  - Micro view for detailed layer inspection
- **Right Pane (Controls/Summary)**: 1/3 width on large screens
  - Execution controls (play/pause, step forward/back)
  - Summary panel with output probabilities
- **Responsive**: Stacks vertically on mobile devices

### 8. Environment Variables
Configured in `.env.example` and `.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
```

The API client reads this via `import.meta.env.VITE_API_BASE_URL` with a fallback to `http://localhost:8000`.

### 9. API Client Helper
`src/api/client.ts` provides:
- Axios instance configured with base URL from environment
- Request timeout (30 seconds)
- Built-in caching for repeated requests
- Error handling with user-friendly messages
- `runModelForward(text: string)` method for model execution
- `clearCache()` utility

`src/api/queries.ts` provides:
- React Query hooks for API integration
- `useModelForwardMutation()` for model execution with cache management

### 10. React Query Integration
- QueryClient configured in `main.tsx`
- Default options set (no refetch on window focus, 1 retry)
- App wrapped in `QueryClientProvider`
- Example mutation hook available

### 11. Routing
Currently single-page application with view-based architecture:
- Main view in `src/views/HomeView.tsx`
- Can easily extend to multi-page with React Router if needed
- View layer separated from main App.tsx for better organization

## 🚀 Getting Started

### Installation
```bash
cd frontend
npm install
```

### Development
```bash
npm run dev
```
Visit http://localhost:5173

### Production Build
```bash
npm run build
```
Output in `dist/` directory

### Code Quality
```bash
npm run lint    # Check code quality
npm run format  # Format code
```

## ✅ Acceptance Criteria Met

1. ✅ `npm install` successfully installs all dependencies
2. ✅ `npm run dev` serves the React app on port 5173
3. ✅ Shows placeholder layout with:
   - Header with title and description
   - Text input area
   - Split-pane layout (macro/micro views + controls/summary)
   - Loading states
   - Error display
   - Responsive grid layout
4. ✅ `npm run lint` passes without errors
5. ✅ API client points to configurable backend URL via environment variables
6. ✅ Path aliases configured for clean imports
7. ✅ All required packages installed (d3, zustand, react-query, framer-motion, TailwindCSS)
8. ✅ Folder structure established (components, views, state, api)

## 📝 Additional Features

- Keyboard shortcuts support (←/→ for step navigation, Space for play/pause)
- Zustand store for global state management
- Framer Motion ready for animations
- D3 utilities for data visualization
- Lucide React icons for UI
- TypeScript with strict type checking
- Modern React patterns (hooks, functional components)
- Proper error boundaries and loading states
- Breadcrumb navigation component
- Multiple inspector components (Attention, MoE, Heatmap)

## 🔧 Next Steps

The frontend scaffold is complete and ready for further development:
- Add more views as needed
- Implement D3 visualizations in components
- Add animations with Framer Motion
- Extend API client with additional endpoints
- Add routing if multi-page navigation is needed
- Implement additional features per requirements
