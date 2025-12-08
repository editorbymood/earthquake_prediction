---
description: Run the interactive Earthquake Prediction Dashboard
---
1. Ensure the training pipeline has been run at least once to generate models:
```bash
./venv/bin/python main.py
```

2. Launch the Streamlit application:
```bash
./venv/bin/streamlit run app.py
```
This will open the dashboard in your default browser (usually at http://localhost:8501).

# Features
- **Global Map**: Visualize recent seismic activity with filtering.
- **Predictor**: Enter custom coordinates and depth to predict magnitude.
- **Analytics**: View model performance and feature importance.
