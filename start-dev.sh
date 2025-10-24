#!/bin/bash

set -e

# Start backend in background
echo "Starting backend on port 8000..."
cd backend
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Start frontend in background
echo "Starting frontend on port 5173..."
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo ""
echo "=========================================="
echo "✓ Services started!"
echo "=========================================="
echo ""
echo "Backend:  http://localhost:8000 (PID: $BACKEND_PID)"
echo "Frontend: http://localhost:5173 (PID: $FRONTEND_PID)"
echo ""
echo "Logs:"
echo "  Backend:  tail -f backend.log"
echo "  Frontend: tail -f frontend.log"
echo ""
echo "To stop services:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Saving PIDs to .pids file..."
echo "$BACKEND_PID $FRONTEND_PID" > .pids

# Wait for user interrupt
echo "Press Ctrl+C to stop all services..."
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm .pids; exit" INT

wait
