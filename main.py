# Import necessary libraries
import os
import pandas as pd
from sklearn.impute import SimpleImputer

# Function to load data
def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully!")
        return df
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: Could not parse the file. Check the file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Function to display basic information about the dataset
def display_dataset_info(df):
    """Displays basic information about the dataset."""
    print("\nBasic Information about the Dataset:")
    print(df.info())
    print("\nFirst Few Rows of the Dataset:")
    print(df.head())

# Function to handle missing values and duplicates
def handle_missing_values_and_duplicates(df):
    """Handles missing values and duplicates in the dataset."""
    # Check for null values
    print("\nNull Values in Each Column:")
    print(df.isnull().sum())

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nNumber of Duplicate Rows: {duplicates}")

    # Handle duplicates
    if duplicates > 0:
        df.drop_duplicates(inplace=True)
        print(f"\nRemoved {duplicates} duplicate rows.")
    else:
        print("\nNo duplicate rows found.")

    # Handle missing values
    # Use median imputation for numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # Check for null values after imputation
    print("\nNull Values in Each Column After Imputation:")
    print(df.isnull().sum())

    return df

# Main function
def main():
    file_path = "microbiome.csv"
    df = load_data(file_path)
    
    if df is not None:
        display_dataset_info(df)
        df = handle_missing_values_and_duplicates(df)
        display_dataset_info(df)

if __name__ == "__main__":
    main()
