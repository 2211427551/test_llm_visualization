#!/bin/bash

echo "Verifying Model Execution Visualizer Setup"
echo "=========================================="
echo ""

# Check frontend structure
echo "Checking frontend structure..."
FRONTEND_FILES=(
  "frontend/package.json"
  "frontend/tsconfig.json"
  "frontend/vite.config.ts"
  "frontend/src/App.tsx"
  "frontend/src/main.tsx"
  "frontend/src/store/executionStore.ts"
  "frontend/src/api/client.ts"
  "frontend/src/components/TextInput.tsx"
  "frontend/src/components/ExecutionControls.tsx"
  "frontend/src/components/MacroView.tsx"
  "frontend/src/components/MicroView.tsx"
  "frontend/src/components/SummaryPanel.tsx"
  "frontend/src/hooks/useKeyboardShortcuts.ts"
  "frontend/src/hooks/usePlayback.ts"
)

for file in "${FRONTEND_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✓ $file"
  else
    echo "  ✗ MISSING: $file"
  fi
done

echo ""
echo "Checking backend structure..."
BACKEND_FILES=(
  "backend/pyproject.toml"
  "backend/app/main.py"
  "backend/app/models.py"
  "backend/app/api/routes.py"
  "backend/tests/test_api.py"
)

for file in "${BACKEND_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✓ $file"
  else
    echo "  ✗ MISSING: $file"
  fi
done

echo ""
echo "Checking documentation..."
DOC_FILES=(
  "README.md"
  "USAGE.md"
  "frontend/README.md"
  "backend/README.md"
)

for file in "${DOC_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "  ✓ $file"
  else
    echo "  ✗ MISSING: $file"
  fi
done

echo ""
echo "Checking Python syntax..."
cd backend
python3 -m py_compile app/main.py app/models.py app/api/routes.py 2>&1
if [ $? -eq 0 ]; then
  echo "  ✓ Python files compile successfully"
else
  echo "  ✗ Python syntax errors detected"
fi
cd ..

echo ""
echo "=========================================="
echo "Verification complete!"
echo ""
echo "Next steps:"
echo "  1. Run: ./setup.sh"
echo "  2. Start backend and frontend"
echo "  3. Open http://localhost:5173"
