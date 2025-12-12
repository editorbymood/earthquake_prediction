#!/bin/bash

# Start Setup
echo "Starting SeismicAI..."

# 1. Build Frontend
echo "Building Frontend..."
cd frontend
npm install
npm run build
cd ..

# 2. Start Backend (Unified)
echo "Starting Backend (serving Frontend)..."
cd backend/app
../../.venv/bin/uvicorn main:app --reload --port 8000 --host 0.0.0.0 > ../../backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

echo "Backend started with PID $BACKEND_PID"

echo "=================================================="
echo "Project is running on http://localhost:8000"
echo "Logs are in backend.log"
echo "=================================================="

# Wait for user input to exit
read -p "Press Enter to stop servers..."

kill $BACKEND_PID
echo "Server stopped."
