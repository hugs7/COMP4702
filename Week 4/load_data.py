"""
Helper file to load data using pandas
"""

import pandas as pd


def load_data(file_path: str)-> pd.DataFrame:
    df = None

    # If Excel file
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)

    # If CSV file
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)


    return df
