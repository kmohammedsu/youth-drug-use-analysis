import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans the youth_data dataset by handling special codes and preparing variables for analysis.
    """
    # Handle special codes (retain 91 and 93, replace 97 and 98 with NaN)
    special_codes = {97: np.nan, 98: np.nan}
    substance_cols = ['IRCIGFM', 'IRALCFY', 'IRMJFY']
    df[substance_cols] = df[substance_cols].replace(special_codes)

    # Create binary target variable for cigarette use
    df['cigarette_use'] = df['IRCIGFM'].apply(lambda x: 0 if x in [91, 93] else 1)

    # Create multi-class target variable for marijuana frequency
    df['marijuana_freq'] = pd.cut(
        df['IRMJFY'], 
        bins=[-1, 0, 30, 60, np.inf], 
        labels=['Never', 'Seldom', 'Sometimes', 'Frequent']
    )

    # Drop rows with missing values in key columns
    key_columns = substance_cols + ['ALCYDAYS', 'PARHLPHW', 'SCHFELT', 'INCOME']
    df.dropna(subset=key_columns, inplace=True)

    return df


if __name__ == "__main__":
    # Path to raw data
    raw_data_path = "./data/youth_data.csv"  
    cleaned_data_path = "./data/processed_data.csv"

    df = pd.read_csv(raw_data_path)
    cleaned_df = clean_data(df)
    cleaned_df.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned data saved to {cleaned_data_path}")

