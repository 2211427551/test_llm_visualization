#!/bin/bash

# Test script for Micro Inspectors functionality

set -e

echo "🧪 Testing Micro Inspectors Implementation"
echo "=========================================="
echo ""

# Check if backend is running
echo "1. Checking backend health..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "   ✅ Backend is healthy"
else
    echo "   ❌ Backend is not responding"
    exit 1
fi
echo ""

# Test model forward endpoint
echo "2. Testing model forward endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:8000/model/forward \
    -H "Content-Type: application/json" \
    -d '{"text":"Test attention and MoE"}')

if echo "$RESPONSE" | grep -q "success"; then
    echo "   ✅ Model forward endpoint working"
else
    echo "   ❌ Model forward endpoint failed"
    exit 1
fi
echo ""

# Verify attention data
echo "3. Verifying attention layer data..."
HAS_ATTENTION=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print('attentionData' in data['steps'][1]['layerData'])")
if [ "$HAS_ATTENTION" = "True" ]; then
    echo "   ✅ Attention data present"
    
    # Check attention components
    NUM_HEADS=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data['steps'][1]['layerData']['attentionData']['numHeads'])")
    echo "   • Number of heads: $NUM_HEADS"
    
    HAS_SCORES=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data['steps'][1]['layerData']['attentionData']['attentionScores'] is not None)")
    if [ "$HAS_SCORES" = "True" ]; then
        echo "   • ✅ Attention scores present"
    fi
    
    HAS_MASK=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data['steps'][1]['layerData']['attentionData']['sparsityMask'] is not None)")
    if [ "$HAS_MASK" = "True" ]; then
        echo "   • ✅ Sparsity mask present"
    fi
    
    HAS_QKV=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); ad = data['steps'][1]['layerData']['attentionData']; print(ad['queryMatrix'] is not None and ad['keyMatrix'] is not None and ad['valueMatrix'] is not None)")
    if [ "$HAS_QKV" = "True" ]; then
        echo "   • ✅ Q/K/V matrices present"
    fi
else
    echo "   ❌ Attention data missing"
    exit 1
fi
echo ""

# Verify MoE data
echo "4. Verifying MoE layer data..."
HAS_MOE=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print('moeData' in data['steps'][2]['layerData'])")
if [ "$HAS_MOE" = "True" ]; then
    echo "   ✅ MoE data present"
    
    # Check MoE components
    NUM_EXPERTS=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data['steps'][2]['layerData']['moeData']['numExperts'])")
    echo "   • Number of experts: $NUM_EXPERTS"
    
    TOP_K=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data['steps'][2]['layerData']['moeData']['topK'])")
    echo "   • Top-K selection: $TOP_K"
    
    HAS_GATING=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data['steps'][2]['layerData']['moeData']['gatingWeights'] is not None)")
    if [ "$HAS_GATING" = "True" ]; then
        echo "   • ✅ Gating weights present"
    fi
    
    HAS_EXPERTS=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data['steps'][2]['layerData']['moeData']['selectedExperts'] is not None)")
    if [ "$HAS_EXPERTS" = "True" ]; then
        echo "   • ✅ Selected experts present"
    fi
    
    HAS_ACTIVATIONS=$(echo "$RESPONSE" | python3 -c "import json, sys; data = json.load(sys.stdin); print(data['steps'][2]['layerData']['moeData']['expertActivations'] is not None)")
    if [ "$HAS_ACTIVATIONS" = "True" ]; then
        echo "   • ✅ Expert activations present"
    fi
else
    echo "   ❌ MoE data missing"
    exit 1
fi
echo ""

# Check frontend build
echo "5. Checking frontend build..."
if [ -f "frontend/dist/index.html" ]; then
    echo "   ✅ Frontend build exists"
else
    echo "   ⚠️  Frontend not built (run 'npm run build' in frontend/)"
fi
echo ""

# Check if frontend is running
echo "6. Checking frontend server..."
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "   ✅ Frontend is running"
else
    echo "   ⚠️  Frontend not running (run 'npm run dev' in frontend/)"
fi
echo ""

echo "=========================================="
echo "✅ All Micro Inspector tests passed!"
echo ""
echo "To test manually:"
echo "1. Open http://localhost:5173 in your browser"
echo "2. Enter text: 'Test attention and MoE'"
echo "3. Click 'Run'"
echo "4. Navigate to Step 2 (Multi-Head Attention)"
echo "5. Explore attention scores and Q/K/V matrices"
echo "6. Navigate to Step 3 (Mixture of Experts)"
echo "7. View gating weights and expert routing"
echo ""
