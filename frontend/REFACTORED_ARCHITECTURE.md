# Refactored Frontend Architecture

## Overview

The frontend has been completely refactored to create a professional-grade Transformer visualization tool, inspired by [transformer-explainer](https://github.com/poloclub/transformer-explainer) and [llm-viz](https://bbycroft.net/llm).

## Tech Stack

### Core Framework
- **Next.js 16** - React framework with App Router
- **React 19** - Latest React with concurrent features
- **TypeScript** - Type-safe development

### Visualization Libraries
- **Three.js** - 3D matrix and computation visualization
- **@react-three/fiber** - React renderer for Three.js
- **@react-three/drei** - Useful helpers for R3F
- **D3.js** - 2D data visualization and SVG manipulation

### Animation
- **GSAP** - Professional animation timeline system
- **Framer Motion** - React animation library for smooth transitions

### UI Components
- **shadcn/ui** - High-quality, accessible UI components
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icon system

### Math & Formulas
- **KaTeX** - Fast math typesetting for displaying formulas

### State Management
- **Zustand** - Lightweight state management

## Project Structure

```
frontend/src/
├── app/                              # Next.js App Router pages
│   ├── layout.tsx                    # Root layout with fonts
│   ├── page.tsx                      # Home page
│   ├── transformer-viz/              # New visualization page
│   └── globals.css                   # Global styles + KaTeX
├── components/
│   ├── layout/                       # Layout components
│   │   ├── AppLayout.tsx            # Main app container
│   │   ├── Header.tsx               # Top navigation
│   │   ├── Sidebar.tsx              # Control sidebar
│   │   └── ScrollytellingLayout.tsx # Scroll-based narrative
│   ├── visualization/                # Visualization components
│   │   ├── VisualizationCanvas.tsx  # Main canvas wrapper
│   │   ├── attention/
│   │   │   └── AttentionMatrix.tsx  # 2D attention heatmap
│   │   ├── matrix/
│   │   │   └── Matrix3D.tsx         # 3D matrix visualization
│   │   ├── embedding/               # Token embedding viz
│   │   ├── moe/                     # MoE components
│   │   └── shared/                  # Shared viz utilities
│   ├── controls/
│   │   └── PlaybackControls.tsx     # Animation playback UI
│   ├── explanation/
│   │   └── FormulaDisplay.tsx       # Mathematical formulas
│   └── ui/                          # shadcn/ui components
│       ├── button.tsx
│       ├── card.tsx
│       ├── slider.tsx
│       ├── tabs.tsx
│       └── tooltip.tsx
├── lib/
│   └── visualization/               # Visualization utilities
│       ├── colors.ts                # Color system
│       ├── animation.ts             # GSAP animation controller
│       ├── d3-helpers.ts            # D3.js utilities
│       ├── three-helpers.ts         # Three.js utilities
│       └── utils.ts                 # General utilities
├── stores/
│   └── playback-store.ts            # Animation playback state
├── hooks/                           # Custom React hooks
└── types/                           # TypeScript definitions
```

## Key Features

### 1. 3D Matrix Visualization

Using Three.js and React Three Fiber for immersive 3D visualization of matrices:
- Interactive rotation, zoom, and pan
- Real-time highlighting on hover
- Smooth animations
- Configurable camera and lighting

**Usage:**
```tsx
import { Matrix3D } from '@/components/visualization/matrix/Matrix3D';

<Matrix3D
  data={matrixData}
  title="Attention Weights"
  showValues={true}
  interactive={true}
  onCellClick={(i, j, value) => console.log(value)}
/>
```

### 2. Advanced Animation System

GSAP-based timeline animation controller with full playback controls:
- Play, pause, step forward/back
- Speed control (0.25x - 3x)
- Progress scrubbing
- Step markers and labels

**Usage:**
```tsx
import { AnimationController } from '@/lib/visualization/animation';

const controller = new AnimationController();
controller
  .addMatrixMultiplication(matrixA, matrixB, result)
  .addFadeIn(elements)
  .play();
```

### 3. Interactive 2D Visualizations

D3.js-powered 2D visualizations with rich interactions:
- Attention heatmaps with hover effects
- Connection lines showing data flow
- Animated entry transitions
- Responsive axis labels

**Usage:**
```tsx
import { AttentionMatrix } from '@/components/visualization/attention/AttentionMatrix';

<AttentionMatrix
  attentionWeights={weights}
  tokens={tokens}
  title="Self-Attention"
  showValues={false}
  onHover={(i, j, value) => console.log(value)}
/>
```

### 4. Playback Controls

Comprehensive playback system for stepping through computation:
- Visual progress bar with step markers
- Current step information
- Speed adjustment
- Auto-advance with animation loop

**State Management:**
```tsx
import { usePlaybackStore } from '@/stores/playback-store';

const { 
  isPlaying, 
  currentStep, 
  play, 
  pause, 
  nextStep 
} = usePlaybackStore();
```

### 5. Mathematical Formulas

KaTeX integration for beautiful formula rendering:
- Display mode and inline mode
- Pre-defined common formulas
- Variable explanations
- LaTeX syntax support

**Usage:**
```tsx
import { FormulaDisplay, FORMULAS } from '@/components/explanation/FormulaDisplay';

<FormulaDisplay
  formula={FORMULAS.attention}
  explanation="Scaled dot-product attention mechanism"
  variables={{
    'Q': 'Query matrix',
    'K': 'Key matrix',
    'V': 'Value matrix',
  }}
/>
```

### 6. Scrollytelling Layout

Scroll-based narrative visualization inspired by transformer-explainer:
- Fixed visualization with scrolling content
- Smooth opacity and position transitions
- Step-by-step explanations
- Progress indicators

**Usage:**
```tsx
import { ScrollytellingLayout } from '@/components/layout/ScrollytellingLayout';

<ScrollytellingLayout
  sections={[
    { id: '1', title: 'Step 1', content: <div>...</div> },
    { id: '2', title: 'Step 2', content: <div>...</div> },
  ]}
  visualization={(scrollProgress) => <YourViz progress={scrollProgress} />}
/>
```

## Design System

### Colors

Following llm-viz inspired dark theme:

```css
--bg-primary: #0a0a0f
--bg-secondary: #13131a
--bg-tertiary: #1a1a24

--accent-blue: #3b82f6
--accent-purple: #a855f7
--accent-pink: #ec4899
--accent-cyan: #06b6d4

--viz-embedding: #8b5cf6
--viz-attention: #ec4899
--viz-ffn: #06b6d4
--viz-output: #10b981
```

### Typography

- **Body**: Inter (Google Font)
- **Code/Math**: JetBrains Mono (Google Font)
- **Formulas**: KaTeX fonts

### Components

Using shadcn/ui for consistent, accessible components:
- Buttons with variants
- Cards with hover effects
- Sliders for controls
- Tabs for view switching
- Tooltips for information

## Getting Started

### Installation

All dependencies are already installed. If you need to reinstall:

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

Visit `http://localhost:3000/transformer-viz` for the new visualization.

### Build

```bash
npm run build
npm start
```

## Component Examples

### Creating a Custom Visualization

```tsx
'use client';

import { usePlaybackStore } from '@/stores/playback-store';
import { Matrix3D } from '@/components/visualization/matrix/Matrix3D';

export function MyVisualization() {
  const { currentStep } = usePlaybackStore();
  
  // Generate data based on current step
  const data = generateDataForStep(currentStep);
  
  return (
    <div className="space-y-6">
      <Matrix3D
        data={data}
        title={`Step ${currentStep + 1}`}
        interactive={true}
      />
    </div>
  );
}
```

### Adding Animation

```tsx
import { useEffect, useRef } from 'react';
import { AnimationController } from '@/lib/visualization/animation';

export function AnimatedComponent() {
  const containerRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<AnimationController>();
  
  useEffect(() => {
    if (!containerRef.current) return;
    
    const controller = new AnimationController();
    const elements = containerRef.current.querySelectorAll('.animate-me');
    
    controller
      .addFadeIn(Array.from(elements))
      .addLabel('shown')
      .addPause(1)
      .play();
    
    controllerRef.current = controller;
    
    return () => controller.clear();
  }, []);
  
  return <div ref={containerRef}>...</div>;
}
```

## Performance Considerations

### 3D Rendering
- Use `useMemo` for expensive computations
- Implement LOD (Level of Detail) for large matrices
- Consider using instanced meshes for many objects

### D3 Updates
- Use D3's data join pattern for efficient updates
- Transition durations should be kept under 1 second
- Use `requestAnimationFrame` for smooth animations

### State Management
- Keep playback state separate from visualization data
- Use Zustand's selective subscriptions
- Memoize derived state

## Browser Compatibility

- **Chrome/Edge**: Full support
- **Firefox**: Full support
- **Safari**: Full support (WebGL 2.0 required)
- **Mobile**: Limited (3D visualizations may lag)

## Future Enhancements

- [ ] WebGL shader-based visualizations
- [ ] Export animations as video
- [ ] VR/AR support with WebXR
- [ ] Real-time model inference with transformers.js
- [ ] Collaborative features (multi-user)
- [ ] Custom model upload and visualization
- [ ] Performance profiling dashboard

## References

- [transformer-explainer](https://github.com/poloclub/transformer-explainer) - Scrollytelling and interaction design
- [llm-viz](https://bbycroft.net/llm) - 3D visualization and technical accuracy
- [Three.js Documentation](https://threejs.org/docs/)
- [D3.js Documentation](https://d3js.org/)
- [GSAP Documentation](https://gsap.com/docs/v3/)
- [shadcn/ui Components](https://ui.shadcn.com/)

## License

MIT License - See main repository LICENSE file

---

**Note**: This refactor represents a complete architectural overhaul focused on creating a professional, educational, and interactive visualization tool for understanding Transformer models.
