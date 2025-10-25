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

- React 18
- TypeScript
- Vite
- Zustand (state management)
- Framer Motion (animations)
- Tailwind CSS
- Axios
