# Tech Minds: Predicting Mental Health Support Needs in the Tech Industry

## Project Overview
This project aims to predict mental health support needs among professionals in the tech industry using survey data and machine learning. The goal is to identify individuals who may benefit from mental health support, enabling organizations to provide timely assistance.

## Project Structure
```
├── app/                # Application code (e.g., dashboards, APIs)
├── data/               # Data storage
│   ├── raw/            # Raw data files
│   └── processed/      # Processed data files
├── models/             # Saved models
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── outputs/            # Generated figures and reports
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
- Run data processing and model training scripts in `src/`.
- Explore data and results in the `notebooks/` directory.
- Launch the dashboard or app from `app/app.py` (if implemented).

## Data
The main dataset is located at `data/raw/survey.csv`.

## License
MIT License.