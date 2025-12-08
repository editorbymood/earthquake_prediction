---
description: Run the complete Earthquake Prediction Pipeline
---
1. Install dependencies (if not already done):
```bash
./venv/bin/pip install -r requirements.txt
```

2. Run the main training pipeline:
```bash
./venv/bin/python main.py
```
This will:
- Download real earthquake data from USGS (Magnitude 4.5+, Last 5 years).
- Run Genetic Algorithm to select optimal features.
- Train Ensemble Models (RandomForest + XGBoost).
- Train Deep Learning Model (TensorFlow/Keras).
- Generate plots in `plots/` and results in `results/`.

3. Run the Web Application (Optional):
```bash
./venv/bin/streamlit run app.py
```
This opens a dashboard to view results and make predictions.
