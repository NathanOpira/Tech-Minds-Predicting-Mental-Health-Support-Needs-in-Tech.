# Importing required libraries.
import pandas as pd

def load_clean_data(path='../data/processed/mental_health_cleaned.csv'):
    """
    Loads the cleaned mental health dataset.

    Parameters:
        path (str): File path to the cleaned dataset.

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    df = pd.read_csv(path)
    return df