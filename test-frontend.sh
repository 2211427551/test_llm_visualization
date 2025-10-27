#!/bin/bash

echo "🧪 Testing Frontend Setup..."
echo ""

cd frontend

echo "1️⃣ Checking dependencies..."
if [ ! -d "node_modules" ]; then
  echo "❌ node_modules not found. Run: npm install"
  exit 1
fi
echo "✅ Dependencies installed"

echo ""
echo "2️⃣ Checking file structure..."
REQUIRED_FILES=(
  "src/app/page.tsx"
  "src/app/layout.tsx"
  "src/app/globals.css"
  "src/components/InputModule.tsx"
  "src/components/ControlPanel.tsx"
  "src/components/VisualizationCanvas.tsx"
  "src/components/ExplanationPanel.tsx"
  "src/store/visualizationStore.ts"
  "src/services/api.ts"
  "src/types/index.ts"
  "package.json"
  "tsconfig.json"
  ".env.local"
)

for file in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "❌ Missing: $file"
    exit 1
  fi
done
echo "✅ All required files present"

echo ""
echo "3️⃣ Checking TypeScript..."
npx tsc --noEmit
if [ $? -eq 0 ]; then
  echo "✅ TypeScript check passed"
else
  echo "❌ TypeScript check failed"
  exit 1
fi

echo ""
echo "4️⃣ Checking ESLint..."
npm run lint
if [ $? -eq 0 ]; then
  echo "✅ ESLint check passed"
else
  echo "❌ ESLint check failed"
  exit 1
fi

echo ""
echo "5️⃣ Building production..."
npm run build
if [ $? -eq 0 ]; then
  echo "✅ Production build successful"
else
  echo "❌ Production build failed"
  exit 1
fi

echo ""
echo "🎉 All tests passed!"
echo ""
echo "To start the development server:"
echo "  cd frontend && npm run dev"
echo ""
echo "To start the production server:"
echo "  cd frontend && npm start"
