import os
import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configuration for logging
logging.basicConfig(filename='preprocessing_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads data from a CSV file, handling potential errors."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        logging.error(f"Error loading data: {e}")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
        return None

def display_dataset_info(df, title="Dataset Information"):
    """Displays basic information about the dataset."""
    print(f"\n{title}:")
    print(df.info())
    print("\nFirst Few Rows:")
    print(df.head())


def handle_missing_values_and_duplicates(df):
    """Handles missing values and duplicates."""
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        logging.info(f"Removed {duplicates} duplicate rows.")
    else:
        logging.info("No duplicate rows found.")

    # Missing Values
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for cols, strategy in [(numerical_cols, 'median'), (categorical_cols, 'most_frequent')]:
        imputer = SimpleImputer(strategy=strategy)
        df[cols] = imputer.fit_transform(df[cols])

    logging.info(f"Null values after imputation:\n{df.isnull().sum()}")
    return df


def transform_data(df):
    """Performs data transformations."""
    species_cols = [col for col in df.columns if col.startswith('Species')]

    # Relative Abundance
    df[species_cols] = df[species_cols].div(df[species_cols].sum(axis=1), axis=0)

    # Scaling
    scaler = StandardScaler()
    df[species_cols] = scaler.fit_transform(df[species_cols])

    return df


def main():
    file_path = "microbiome.csv"
    df = load_data(file_path)

    if df is not None:
        try:
            display_dataset_info(df, "Original Data")
            df = handle_missing_values_and_duplicates(df)
            df = transform_data(df)
            display_dataset_info(df, "Preprocessed Data")
            df.to_csv("cleaned_microbiome_data.csv", index=False)
            logging.info("Data preprocessing completed successfully.")
            print("Preprocessed data saved to cleaned_microbiome_data.csv")
        except Exception as e:
            logging.exception(f"An error occurred during preprocessing: {e}")


if __name__ == "__main__":
    main()
