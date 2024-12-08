#import libraries 

import os
import pandas as pd
import google.generativeai as genai
import time

#eda 
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
