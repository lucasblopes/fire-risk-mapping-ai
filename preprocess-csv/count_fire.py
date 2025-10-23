import pandas as pd


def check_fire_counts(file_path):
    """
    Checks the number of 'is_fire' instances for both 1 and 0 in a CSV file.

    Args:
        file_path (str): The path to the CSV file.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, delimiter=";")

        # Check if the 'is_fire' column exists
        if "is_fire" not in df.columns:
            print(f"Error: 'is_fire' column not found in {file_path}")
            return

        # Get the counts of 1s (fire) and 0s (non-fire)
        fire_counts = df["is_fire"].value_counts()

        # Check if both classes are present
        if 1 in fire_counts.index:
            fire_count = fire_counts.loc[1]
        else:
            fire_count = 0

        if 0 in fire_counts.index:
            non_fire_count = fire_counts.loc[0]
        else:
            non_fire_count = 0

        # Print the results
        print(f"Checking file: {file_path}")
        print(f"Number of 'is_fire' = 1 (Fire): {fire_count}")
        print(f"Number of 'is_fire' = 0 (Non-fire): {non_fire_count}")
        print(f"Total instances: {len(df)}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Specify the paths to your training and testing CSV files
    training_file = "iberfire_train.csv"
    testing_file = "iberfire_test.csv"
    # testing_file = "test.csv"

    # Check the training file
    check_fire_counts(training_file)

    print("-" * 30)

    # Check the testing file
    check_fire_counts(testing_file)
