import pandas as pd

# Define the names for the input and output files
input_filename = "iberfire_old_clean.csv"
train_filename = "iberfire_train.csv"
test_filename = "iberfire_test.csv"

print(f"Reading the source file: {input_filename}...")

try:
    # Read the original CSV into a pandas DataFrame
    df = pd.read_csv(input_filename)

    # --- Data Splitting ---
    # The 'time' column is used to identify the year.
    # We first convert it to a proper datetime format to safely extract the year.
    print("Converting 'time' column to datetime objects...")
    df["time"] = pd.to_datetime(df["time"])

    # Create the test set: select all rows where the year is 2024
    print("Creating the test set for the year 2024...")
    test_df = df[df["time"].dt.year == 2024].copy()

    # Create the training set: select all rows where the year is NOT 2024
    print("Creating the training set for all other years...")
    train_df = df[df["time"].dt.year != 2024].copy()

    # --- Saving the Files ---
    # Save the training DataFrame to a new CSV file
    # index=False prevents pandas from writing the DataFrame index as a column
    print(f"Saving training data to {train_filename}...")
    train_df.to_csv(train_filename, index=False)

    # Save the test DataFrame to another CSV file
    print(f"Saving test data to {test_filename}...")
    test_df.to_csv(test_filename, index=False)

    print("\n✅ Process completed successfully!")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

except FileNotFoundError:
    print(
        f"❌ Error: The file '{input_filename}' was not found in the current directory."
    )
except KeyError:
    print(
        "❌ Error: A 'time' column was not found in the CSV file. Please check the column names."
    )
except Exception as e:
    print(f"An unexpected error occurred: {e}")
