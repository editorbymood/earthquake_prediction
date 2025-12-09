# SeismicAI: Earthquake Prediction System

A professional AI/ML-powered earthquake prediction and monitoring system.

## 🏗 Architecture

This project uses a modern separation of concerns:

- **Frontend**: React + Vite + TailwindCSS (Located in `/frontend`)
- **Backend**: FastAPI + Python (Located in `/backend`)
- **AI/ML Core**: Shared logic, models, and utilities (Located in `/backend/src`)

## 🚀 Getting Started

To run the entire full-stack application with one command:

```bash
./start.sh
```

This will start:
- Backend API at `http://localhost:8000`
- Frontend Dashboard at `http://localhost:5173`

## 📁 Directory Structure

```text
/
├── backend/            # Python FastAPI Backend
│   ├── app/            # API Endpoints
│   └── src/            # Core Business Logic & Models
├── frontend/           # React Frontend
├── data/               # Datasets
├── results/            # Model Training Results
└── start.sh            # Universal Launch Script
```

## 🛠 Tech Stack

- **Frontend**: React, TailwindCSS, Framer Motion, Leaflet, Recharts
- **Backend**: FastAPI, Uvicorn, Pandas, NumPy, Scikit-Learn
- **Models**: Random Forest, XGBoost (Ensemble), Deep Learning

## 📝 Usage

1.  **Dashboard**: View global earthquake data stats and interactive map.
2.  **Predictor**: Enter geophysical parameters to predict earthquake magnitude using our trained AI models.
