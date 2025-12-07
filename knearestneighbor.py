import pandas as pd
import os

# -------------------------------------------------------
# Data can be found at  https://www.kaggle.com/datasets/joebeachcapital/natural-gas-prices
# -------------------------------------------------------

try:
    df = pd.read_csv("daily_csv.csv")
    print("\nFile loaded successfully!\n")

except FileNotFoundError:
    print(f"ERROR: The file was not found at: {csv_path}")
    print("Check the subfolder name, filename, and spelling.")
    exit()

except pd.errors.EmptyDataError:
    print("ERROR: The file exists but is empty.")
    exit()

except pd.errors.ParserError:
    print("ERROR: The file cannot be parsed as a valid CSV.")
    print("Check for mismatched quotes or bad formatting.")
    exit()

except Exception as e:
    print("An unexpected error occurred while loading the file:")
    print(e)
    exit()

# -------------------------------------------------------
# Display first few rows
# -------------------------------------------------------
print("Preview of the loaded data:")
print(df.head())
print(df.describe())
# -------------------------------------------------------
# Warn the user about missing values (NaN)
# -------------------------------------------------------
missing_total = df.isna().sum().sum()

if missing_total > 0:
    print("\nWARNING: Missing values detected in the dataset!")
    print(df.isna().sum())  # Missing count per column
    print(f"Total missing values = {missing_total}")
else:
    print("\nNo missing values found in the dataset.")

# Optional: remind the student what NaN means
print("\nNOTE: 'NaN' means 'Not a Number' and represents missing or invalid data.\n")
