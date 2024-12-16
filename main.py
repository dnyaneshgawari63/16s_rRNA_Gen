# Import libraries
import os
import pandas as pd
import google.generativeai as genai
import time
from sklearn.impute import SimpleImputer

# Load the CSV file
try:
    df = pd.read_csv("microbiome.csv")
    print("Data loaded successfully!")
except FileNotFoundError:
    print("Error: microbiome.csv not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print("Error: microbiome.csv is empty.")
except pd.errors.ParserError:
    print("Error: Could not parse microbiome.csv. Check the file format.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Display basic information about the dataset
print("\nBasic Information about the Dataset:")
print(df.info())

# Display the first few rows of the dataset
print("\nFirst Few Rows of the Dataset:")
print(df.head())

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

# Display basic information about the dataset after handling missing values and duplicates
print("\nBasic Information about the Dataset After Handling Missing Values and Duplicates:")
print(df.info())

# Display the first few rows of the dataset after handling missing values and duplicates
print("\nFirst Few Rows of the Dataset After Handling Missing Values and Duplicates:")
print(df.head())

# Continue with further analysis or processing as needed
