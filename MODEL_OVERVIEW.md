# Model Overview Component

## Overview

The `ModelOverview` component provides a high-level architectural visualization of the Transformer model, rendering the complete layer hierarchy as an interactive SVG diagram. This macro view complements the existing list-based `MacroView` and enables users to understand the model structure at a glance.

## Features

### 1. Hierarchical Architecture Visualization

The component renders the model as a flow diagram with:
- **Embedding Layer** at the bottom
- **Encoder Blocks** in the middle (stacked vertically)
  - Each encoder block displays its child layers (Attention, MoE, Feedforward)
  - Multiple encoder layers automatically stack with proper spacing
- **Output Layer** at the top

### 2. Data Contracts and Normalization

**Location**: `frontend/src/utils/modelNormalizer.ts`

The normalizer transforms the backend `ModelRunResponse` into a structured hierarchy:

```typescript
interface NormalizedModel {
  embedding: ModelNode;
  encoderBlocks: ModelNode[];
  output: ModelNode;
  totalParams: number;
  maxDimension: number;
}
```

Functions:
- `normalizeModelResponse()`: Parses API response and groups layers into encoder blocks
- `estimateParamCount()`: Calculates parameter counts based on layer dimensions and type
- `formatParamCount()`: Formats numbers as K/M/B for display
- `getNodeColor()`: Returns color codes for each layer type

### 3. Interactive Selection

**Clicking a node**:
- Sets `selectedLayerId` in global state
- Updates `currentStepIndex` to focus the micro view
- Builds breadcrumb trail (e.g., "Encoder 1 → Multi-Head Attention")
- Highlights the selected node with a distinct border

**Keyboard navigation**:
- Nodes are keyboard-accessible with `tabIndex={0}`
- Press `Enter` or `Space` to select a node
- ARIA labels describe each layer's type and name
- `aria-pressed` indicates selection state

### 4. Color Coding and Legend

Each layer type has a unique color scheme:
- **Embedding**: Blue (`#3b82f6`)
- **Attention**: Purple (`#8b5cf6`)
- **MoE**: Pink (`#ec4899`)
- **Feedforward**: Green (`#10b981`)
- **Normalization**: Amber (`#f59e0b`)
- **Output**: Red (`#ef4444`)

The legend is toggled via a button in the top-right corner, showing all layer types with their colors.

### 5. Summary Statistics

Displayed in the component header:
- Number of encoder blocks
- Total parameter count (formatted as K/M/B)
- Maximum tensor dimension

Each node displays:
- Layer name
- Output shape (e.g., "5 × 64")
- Parameter count

### 6. Responsive Layout

**Window resize handling**:
- Component dimensions update dynamically using `useEffect` and `ResizeObserver` pattern
- Layout recalculates node positions based on container width
- SVG canvas scales to fit all nodes without overlap
- Minimum height ensures visibility even with small viewports

**Adaptive spacing**:
- Node width: 140px
- Node height: 80px
- Vertical spacing: 60px
- Horizontal spacing (for encoder children): 20px
- Padding: 40px

### 7. Animations

**Framer Motion animations**:
- Nodes fade in and scale up on mount (staggered by 0.05s)
- Active layer scales to 1.05x
- Connections draw in with `pathLength` animation
- Hover effect: scale to 1.05x
- Tooltip fades in/out when hovering nodes

### 8. Accessibility

- SVG has `role="img"` and `aria-label="Transformer model architecture diagram"`
- Each node is a keyboard-focusable button (`role="button"`, `tabIndex={0}`)
- ARIA labels describe layer type and name
- `aria-pressed` indicates selection state
- `aria-expanded` on legend toggle button
- High contrast colors meet WCAG guidelines
- Focus indicators visible on keyboard navigation

## Global State Integration

**New state fields in `executionStore`**:
```typescript
selectedLayerId: string | null;
setSelectedLayer: (layerId: string | null) => void;
breadcrumbs: string[];
setBreadcrumbs: (breadcrumbs: string[]) => void;
```

**Selection flow**:
1. User clicks node in ModelOverview
2. `setSelectedLayer(node.id)` stores selection
3. `setCurrentStep(node.stepIndex)` updates active step
4. `setBreadcrumbs([...])` updates navigation trail
5. MicroView updates to show selected layer's details
6. ExecutionControls highlight the current step

## Breadcrumb Integration

**Location**: `frontend/src/components/Breadcrumb.tsx`

Displays navigation path in the UI:
- Shows hierarchical path (e.g., "Encoder 1 → Multi-Head Attention")
- Animated appearance with Framer Motion
- Last item is bold to indicate current selection
- Chevron separators between items

## View Toggle

Users can switch between:
- **Architecture View**: New ModelOverview component with SVG diagram
- **List View**: Original MacroView with vertical layer list

Toggle buttons in `App.tsx`:
```tsx
<button onClick={() => setViewMode('architecture')}>
  Architecture View
</button>
<button onClick={() => setViewMode('list')}>
  List View
</button>
```

## Usage Example

```tsx
import { ModelOverview } from './components/ModelOverview';

function App() {
  return (
    <div>
      <ModelOverview />
    </div>
  );
}
```

The component automatically:
- Reads data from `useExecutionStore()`
- Normalizes the model structure
- Renders the architecture diagram
- Handles user interactions
- Updates global state on selection

## Testing

### Manual Testing Checklist
- [x] Load model data and verify all layers render
- [x] Click on embedding layer → breadcrumb updates
- [x] Click on encoder block child → micro view updates
- [x] Click on output layer → step index changes
- [x] Resize window → layout adapts without overlap
- [x] Use keyboard to navigate nodes (Tab, Enter, Space)
- [x] Hover nodes → tooltip appears with details
- [x] Toggle legend → colors explained
- [x] Active step → corresponding node highlights
- [x] Multiple encoder blocks → stack correctly

### Acceptance Criteria ✅

✅ **Loading mock backend data renders the complete layer hierarchy**
- Normalizer groups layers into embedding, encoder blocks, and output
- All nodes render with correct positions and spacing
- Connections draw between sequential layers

✅ **Clicking a layer triggers selection events stored in global state**
- `selectedLayerId` updates on click
- `currentStepIndex` syncs with selected node's step
- `breadcrumbs` array builds navigation path
- MicroView focuses on selected layer

✅ **Layout adapts to window resizing without visual overlap**
- Resize handler updates dimensions on `window.resize` event
- Node positions recalculate based on container width
- SVG height scales to fit all nodes
- Minimum spacing prevents overlap

## Future Enhancements

- Zoom and pan controls for large models
- Drag-and-drop to reorder layers (visualization only)
- Export diagram as PNG/SVG
- Minimap for navigation in large models
- Layer grouping/collapse for deep architectures
- Compare two model architectures side-by-side
- Dark mode support
- Custom color themes
