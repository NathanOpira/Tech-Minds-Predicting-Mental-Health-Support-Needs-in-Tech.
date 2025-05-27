
# Tech Minds: Predicting Mental Health Support Needs in the Tech Industry

## Project Overview
This project predicts whether individuals in the tech industry are likely to seek mental health treatment, using survey data and machine learning. The workflow covers data preprocessing, feature engineering, model training, evaluation, explainability (with SHAP), and an interactive dashboard.

## Project Structure
```
├── app/                # Application code (dashboard, main pipeline)
├── data/               # Data storage
│   ├── raw/            # Raw data files
│   └── processed/      # Processed data files
├── models/             # Saved models (pkl)
├── notebooks/          # Jupyter notebooks for EDA, modeling, explainability, dashboard
├── outputs/            # Generated figures and reports
│   ├── figures/        # Plots (correlation, class distribution, SHAP, etc.)
│   └── reports/        # Model evaluation and project report
├── src/                # Source code (data loading, features, training, explainability)
├── config.yaml         # Project configuration
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup
1. Clone the repository.
2. Install dependencies:
   ```pwsh
   pip install -r requirements.txt
   ```
3. Adjust `config.yaml` as needed for your environment.

## Usage
- Run the main pipeline: `python app/app.py`
- Explore data and results in the `notebooks/` directory (EDA, feature engineering, modeling, explainability, dashboard).
- Launch the dashboard: open and run `notebooks/dashboard.ipynb` in Jupyter, or adapt to a standalone Dash app if needed.

## Data
- Raw survey data: `data/raw/survey.csv`
- Processed/encoded data: `data/processed/mental_health_cleaned.csv`

## Outputs
- Figures: Correlation matrix, class distribution, SHAP plots (in `outputs/figures/`)
- Reports: Model evaluation and project summary (in `outputs/reports/`)
- Trained models: Saved in `models/`

## Key Features
- Data cleaning and feature engineering
- Multiple ML models: Logistic Regression, Random Forest, XGBoost
- Model evaluation with classification metrics and ROC AUC
- Model explainability with SHAP
- Interactive dashboard for predictions and explanations

## License
MIT License