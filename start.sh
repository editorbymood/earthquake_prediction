#!/bin/bash

# Kill running processes on ports
echo "Stopping existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5173 | xargs kill -9 2>/dev/null

# Start Backend
echo "Starting Backend..."
cd backend/app
../../venv/bin/uvicorn main:app --reload --port 8000 --host 0.0.0.0 > ../../backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

echo "Backend started with PID $BACKEND_PID"

# Start Frontend
echo "Starting Frontend..."
cd frontend
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "Frontend started with PID $FRONTEND_PID"

echo "=================================================="
echo "Project is running!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "logs are in backend.log and frontend.log"
echo "=================================================="

# Wait for user input to exit
read -p "Press Enter to stop servers..."

kill $BACKEND_PID
kill $FRONTEND_PID
echo "Servers stopped."
