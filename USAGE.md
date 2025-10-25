# Usage Guide - Model Execution Visualizer

## Quick Start

### 1. Start the Backend

```bash
cd backend
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` and interactive documentation at `http://localhost:8000/docs`.

### 2. Start the Frontend

In a new terminal:

```bash
cd frontend
npm install
npm run dev
```

The application will be available at `http://localhost:5173`.

## Using the Application

### Step 1: Enter Text

1. Type or paste text into the input field (up to 500 characters)
2. Watch the character count and estimated token count update in real-time
3. Click "Run" to send the text to the model

### Step 2: View Execution

Once the model runs, you'll see:

#### Execution Controls Panel
- **Run Button**: Starts processing the text
- **Play/Pause Button (Space)**: Automates stepping through layers
- **Previous/Next Buttons (← →)**: Step backward/forward through layers
- **Progress Slider**: Drag to jump to any step
- **Speed Controls**: Adjust playback speed (0.5x, 1x, 1.5x, 2x)

#### Macro View
- Shows all model layers in sequence
- Current layer is highlighted with a blue border
- Completed layers show in green
- Animated indicator shows data flow between layers
- Each layer displays input/output shapes

#### Micro View
- Detailed view of the current layer
- Shows layer name, ID, and description
- Displays activation and weight matrices (when available)
- Hover over matrix values for exact numbers
- Highlights significant values in blue

#### Summary Panel
- Displays input text and token count
- Shows all tokens as colored badges
- Lists top output probabilities with animated bars
- Model information summary

### Step 3: Navigate

Use keyboard shortcuts for efficient navigation:
- **←** : Previous step
- **→** : Next step
- **Space** : Play/Pause automated playback
- **Home** : Jump to first step
- **End** : Jump to last step

## Features

### Text Input Validation
- Real-time character counting (max 500)
- Token count estimation (~1.3x word count)
- Warning at 450+ characters
- Input disabled while processing

### Error Handling
- Friendly error messages when backend is down
- Validation errors for invalid input
- Dismissible error notifications
- Automatic error state clearing on new run

### Data Caching
- API responses are cached per input text
- Repeated submissions use cached data
- No redundant backend calls

### Truncation Warnings
- Backend truncates tensor data for inputs >100 tokens
- Visual warnings shown in UI
- Yellow badges indicate truncated layers

## Model Architecture

The simulated model includes 6 layers:

1. **Embedding Layer**: Converts tokens to dense vectors (64-dim)
2. **Multi-Head Attention**: Self-attention mechanism
3. **Feed Forward Network**: Position-wise transformation (128-dim)
4. **Layer Normalization**: Training stability
5. **Output Projection**: Projects to output space (32-dim)
6. **Softmax Layer**: Probability distribution

## API Examples

### Run Model Forward Pass

```bash
curl -X POST "http://localhost:8000/model/forward" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### Check Health

```bash
curl http://localhost:8000/health
```

## Customization

### Backend Environment Variables

Create `backend/.env`:
```env
HOST=0.0.0.0
PORT=8000
DEBUG=true
ALLOWED_ORIGINS=http://localhost:5173
```

### Frontend Environment Variables

Create `frontend/.env`:
```env
VITE_API_BASE_URL=http://localhost:8000
```

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
lsof -ti:8000 | xargs kill -9
```

**Poetry not found:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Module not found:**
```bash
cd backend
poetry install --no-cache
```

### Frontend Issues

**npm install fails:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**CORS errors:**
Ensure backend `.env` has `ALLOWED_ORIGINS=http://localhost:5173`

**Build errors:**
```bash
npm run lint
```

## Performance

- Frontend renders smoothly with Framer Motion animations
- Backend truncates large tensors to keep payload <1MB
- Response caching eliminates redundant API calls
- Lazy loading for matrix visualizations

## Accessibility

- Full keyboard navigation support
- ARIA labels on all interactive elements
- Focus management for modal interactions
- High contrast color scheme
- Clear visual feedback for all actions
