# Boston Housing Price Predictor

A small demo application that predicts Boston housing prices using a Decision Tree regressor. This repository contains a Streamlit-based interactive app, the processed dataset, and the code used to train and serve the model.

## Features
- Interactive Streamlit UI with sliders and a clean layout
- Sidebar with model summary and sample data preview
- Top feature importances and dataset statistics
- Trained Decision Tree model tuned with Grid Search

## Getting Started
These instructions will get the project running locally.

Prerequisites
- Python 3.8+ installed
- Recommended: create and activate a virtual environment

Install dependencies

```
pip install -r requirements.txt
```

Run the app

```
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501` (or Streamlit will show the correct URL in the terminal).

## Project Structure
- `app.py` - Main Streamlit application (interactive UI + model inference)
- `data/processed/boston.csv` - Processed dataset used by the app
- `models/` - (optional) directory for saved model artifacts
- `notebooks/` - Exploratory notebooks and experiments
- `src/` - Utility modules (visuals, helpers)

## Notes & Limitations
- This is a demonstration model trained on a small, processed dataset. Predictions should not be used for real-world decisions without additional validation and ethical review.
- The app shows training R² and feature importances to help understand model behavior, but a proper evaluation pipeline (train/test split, cross validation on unseen data) is required for production use.

## Contributing
Contributions, improvements, and bug reports are welcome. Please open an issue or submit a pull request.