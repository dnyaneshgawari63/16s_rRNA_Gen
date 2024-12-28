import os
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import logging

# Config logging
logging.basicConfig(filename='preprocessing_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        logging.error("Error: File not found. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        logging.error("Error: Could not parse the file. Check the file format.")
        return None
    except Exception as e:
        logging.exception(f"An unexpected error occurred during file loading: {e}")
        return None


def display_dataset_info(df):
    """Displays basic information about the dataset."""
    print("\nBasic Information about the Dataset:")
    print(df.info())
    print("\nFirst Few Rows of the Dataset:")
    print(df.head())


def handle_missing_values_and_duplicates(df):
    """Handles missing values and duplicates in the dataset."""
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of Duplicate Rows: {duplicates}")

    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"\nRemoved {duplicates} duplicate rows.")
        logging.info(f"Removed {duplicates} duplicate rows.")
    else:
        print("\nNo duplicate rows found.")
        logging.info("No duplicate rows found.")


    # Handle missing values - improved
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns


    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    #Check for remaining nulls after imputation
    print("\nNull Values After Imputation:")
    print(df.isnull().sum())
    logging.info(f"Null values after imputation:\n{df.isnull().sum()}")

    return df


def transform_data(df):
    """Performs data transformations (relative abundance, scaling, log transformation)."""

    # Separate features for scaling.  Adjust this to exclude non-numeric columns like sample IDs
    species_cols = [col for col in df.columns if col.startswith('Species')]
    features_to_scale = df[species_cols]

    # Relative Abundance calculation.  This assumes the species columns are already counts.
    total_counts = features_to_scale.sum(axis=1)
    relative_abundance = features_to_scale.div(total_counts, axis=0)
    df[species_cols] = relative_abundance

    # Apply scaling (choose one - StandardScaler or MinMaxScaler)
    scaler = StandardScaler()  # Or MinMaxScaler() if you prefer scaling to [0, 1]
    scaled_data = scaler.fit_transform(relative_abundance)
    df[species_cols] = scaled_data

    # Log transformation (optional, be cautious; handle potential errors if values <=0)

    #Example of log transform (only if data is suitable). Handle potential errors.
    #df[species_cols] = np.log1p(df[species_cols])


    return df


def main():
    file_path = "microbiome.csv"
    df = load_data(file_path)

    if df is not None:
        try:
            display_dataset_info(df)
            df = handle_missing_values_and_duplicates(df)
            df = transform_data(df)
            display_dataset_info(df)
            logging.info("Data preprocessing completed successfully.")
            #Save the cleaned and transformed data to a new file.
            df.to_csv("cleaned_microbiome_data.csv", index=False)
            print("Preprocessed data saved to cleaned_microbiome_data.csv")
        except Exception as e:
            logging.exception(f"An error occurred during preprocessing: {e}")


if __name__ == "__main__":
    main()
