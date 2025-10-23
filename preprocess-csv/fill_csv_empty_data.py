import pandas as pd

# Replace with the path to your CSV file
csv_file = "./iberfire_testing_dataset.csv"

# Read CSV
df = pd.read_csv(csv_file)

print("Before filling missing values:")
print(df.isnull().sum())  # show how many NaNs per column

# Fill missing values with mean (only for numeric columns)
df_filled = df.fillna(df.mean(numeric_only=True))

print("\nAfter filling missing values:")
print(df_filled.isnull().sum())

# Optionally save the new dataset
df_filled.to_csv("iberfire_testing_balanced_clean.csv", index=False)
print("\nSaved cleaned dataset as 'iberfire_preprocessed_balanced_clean.csv'")
