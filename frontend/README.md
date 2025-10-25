# Frontend Application

React + TypeScript frontend for the Model Execution Visualizer.

## Features

- **Text Input**: Enter text with real-time token count feedback and validation
- **Execution Controls**: Run, step forward/backward, and automated playback
- **Macro View**: High-level visualization of model architecture with animated layer highlights
- **Micro View**: Detailed layer-by-layer inspection with matrix visualizations
- **Summary Panel**: Output probabilities and token predictions
- **Keyboard Shortcuts**: 
  - `←` / `→`: Previous / Next step
  - `Space`: Play / Pause
  - `Home`: First step
  - `End`: Last step

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file:
```bash
cp .env.example .env
```

3. Run the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Build

```bash
npm run build
```

## Technologies

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Zustand** - State management
- **TanStack Query (React Query)** - API state management
- **Framer Motion** - Animations
- **D3** - Data visualization
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
- **ESLint** - Code linting
- **Prettier** - Code formatting

## Project Structure

```
src/
├── api/          # API client and backend communication
├── components/   # Reusable React components
├── views/        # Page-level view components
├── store/        # Zustand state management
├── hooks/        # Custom React hooks
├── types/        # TypeScript type definitions
└── utils/        # Utility functions
```

## Path Aliases

The project is configured with absolute imports using the `@` prefix:

- `@/` - src root
- `@components/` - src/components
- `@views/` - src/views
- `@api/` - src/api
- `@store/` - src/store
- `@hooks/` - src/hooks
- `@types/` - src/types
- `@utils/` - src/utils

Example:
```typescript
import { HomeView } from '@views/HomeView';
import { runModelForward } from '@api/client';
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier
- `npm run preview` - Preview production build

## Environment Variables

The application uses environment variables for configuration. Copy `.env.example` to `.env` and adjust as needed:

```
VITE_API_BASE_URL=http://localhost:8000
```
