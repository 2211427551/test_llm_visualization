# Transformer Visualization - Complete UI Rewrite

A comprehensive interactive visualization platform for understanding Transformer model architecture and computations, built with modern web technologies.

## 🚀 New Features

### Architecture
- **Next.js 16** with App Router and TypeScript
- **React Three Fiber** for 3D visualizations
- **D3.js** for data-driven visualizations
- **GSAP** for smooth animations
- **shadcn/ui** for beautiful, accessible components
- **Tailwind CSS** for modern styling

### Interactive Learning
- **Scrollytelling**: Scroll-driven narrative learning experience
- **Timeline Controls**: Play, pause, step through computations
- **3D/2D Mixed**: Dynamic visualization modes
- **Module Containers**: Interactive overlays for each component

### New Routes
- `/explore` - Main interactive learning journey
- `/modules` - Individual component exploration
- `/playground` - Experiment with custom inputs
- `/docs` - Complete documentation

## 🎯 Key Components

### Visualization System
- **3D Transformer Architecture**: Interactive 3D model of layers
- **Attention Visualization**: Multi-head attention patterns
- **Embedding Display**: Token and positional encoding
- **MoE Routing**: Expert selection visualization
- **Output Generation**: Probability distributions

### Animation Framework
- **GSAP Timeline**: Smooth, synchronized animations
- **Keyframe System**: Breakpoint-driven navigation
- **Playback Controls**: Speed control, looping, reverse
- **Progress Tracking**: Real-time step indicators

### Design System
- **Color Palette**: Semantic colors for modules
- **Typography**: Consistent font scales
- **Animations**: Smooth transitions and micro-interactions
- **Dark/Light Theme**: Full theme support

## 🛠️ Technical Implementation

### Frontend Architecture
```
src/
├── app/                    # Next.js App Router
│   ├── explore/            # Main scrollytelling page
│   ├── modules/            # Component-specific pages
│   └── layout.tsx          # Root layout
├── components/
│   ├── layout/             # Header, Sidebar
│   ├── visualization/      # 3D canvases
│   ├── scrollytelling/    # Story sections
│   ├── controls/           # Playback controls
│   ├── modules/            # Module containers
│   └── ui/               # shadcn/ui components
├── contexts/              # React contexts
│   ├── VisualizationContext.tsx
│   ├── AnimationContext.tsx
│   └── ThemeContext.tsx
├── hooks/                 # Custom React hooks
│   ├── useScrollProgress.ts
│   ├── useTransformerAPI.ts
│   └── use-toast.ts
├── services/              # API integration
│   └── api.ts
└── lib/                  # Utilities
    └── design-system.ts
```

### State Management
- **VisualizationContext**: Central visualization state
- **AnimationContext**: Timeline and playback state
- **ThemeContext**: Dark/light mode management

### API Integration
- **Type-safe API client** with Zod validation
- **Server-Sent Events** for real-time streaming
- **Caching** for performance optimization
- **Error handling** and retry logic

## 🎨 Design Principles

### Visual Hierarchy
- **Primary content**: 3D visualization canvas
- **Secondary content**: Explanatory text
- **Controls**: Playback and settings
- **Navigation**: Sidebar and header

### Interaction Patterns
- **Scroll-driven**: Progress through content
- **Click-to-explore**: Detailed information on demand
- **Keyboard shortcuts**: Power user navigation
- **Responsive**: Mobile-friendly design

### Performance
- **60 FPS**: Smooth animations
- **Lazy loading**: Code splitting
- **Optimized rendering**: React Three Fiber optimizations
- **Caching**: API response caching

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.8+
- Docker (optional)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd transformer-viz

# Install frontend dependencies
cd frontend
npm install

# Install backend dependencies
cd ../backend
pip install -r requirements.txt

# Start development servers
npm run dev          # Frontend (port 3000)
python -m app.main   # Backend (port 8000)
```

### Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the application
open http://localhost:3000
```

## 📖 Usage Guide

### Interactive Learning
1. Navigate to `/explore`
2. Scroll through the narrative sections
3. Watch visualizations update in real-time
4. Use playback controls for fine-grained control

### Module Exploration
1. Visit `/modules` for component-specific views
2. Click on module containers for details
3. Interact with 3D visualizations
4. Experiment with different settings

### Custom Experiments
1. Go to `/playground`
2. Input custom text
3. Adjust model parameters
4. Run computations step-by-step

## 🔧 Configuration

### Model Settings
```typescript
const config = {
  n_vocab: 50257,    // Vocabulary size
  n_embd: 768,        // Embedding dimension
  n_layer: 6,          // Number of layers
  n_head: 8,          // Number of attention heads
  max_seq_len: 512,    // Maximum sequence length
}
```

### Visualization Settings
```typescript
const settings = {
  animationSpeed: 1.0,
  showConnections: true,
  showValues: false,
  colorScheme: 'default',
  viewMode: 'mixed',
}
```

## 🧪 Testing

```bash
# Frontend tests
cd frontend
npm run test

# Backend tests
cd backend
python -m pytest

# Type checking
npm run type-check

# Linting
npm run lint
```

## 📊 Performance Metrics

- **First Contentful Paint**: < 1.5s
- **Largest Contentful Paint**: < 2.5s
- **Cumulative Layout Shift**: < 0.1
- **First Input Delay**: < 100ms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Inspired by transformer-explainer and llm-viz
- Built with modern web technologies
- Community-driven development

---

## 🔄 Migration from Old UI

The old UI has been completely replaced. Key changes:

### Removed
- Legacy component structure
- Old visualization system
- Previous routing system
- Outdated styling approach

### Added
- Modern React patterns
- 3D visualization capabilities
- Scrollytelling framework
- Comprehensive animation system

### Breaking Changes
- Route structure updated
- Component APIs changed
- State management refactored
- Styling system replaced

For migration assistance, see the migration guide in `/docs/migration.md`.