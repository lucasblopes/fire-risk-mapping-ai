import pandas as pd
import xarray as xr
import numpy as np
import time


# Helper functions remain the same as they operate row-by-row
def assign_popdens(row):
    """Selects the correct population density column based on the row's year."""
    year = row["year"]
    col = f"popdens_{year}"
    # Check if the column exists in the row's index
    return row[col] if col in row.index else np.nan


def assign_clc(row):
    """Selects the correct Corine Land Cover data based on the nearest valid year."""
    year = row["year"]
    if year < 2012:
        return row["CLC_2006_forest_proportion"]
    elif year < 2018:
        return row["CLC_2012_forest_proportion"]
    else:
        return row["CLC_2018_forest_proportion"]


# Main execution block
if __name__ == "__main__":
    print("⚙️ Starting in-memory data processing with Pandas...")

    start_time = time.time()

    # --- Step 1: Load Full Dataset into Memory ---
    print("➡️ Step 1/7: Loading IberFire.nc dataset...")
    # Load the entire dataset without chunking
    ds = xr.open_dataset("IberFire.nc")
    print("    Dataset loaded.")

    # --- Step 2: Convert to Pandas DataFrame ---
    print("➡️ Step 2/7: Converting Xarray Dataset to Pandas DataFrame...")
    # to_dataframe() loads the data directly into memory
    df = ds.to_dataframe()
    df = df.reset_index()
    print("    Conversion complete.")

    # --- Step 3: Filter Geospatial Area ---
    print("➡️ Step 3/7: Filtering for mainland Spain...")
    df = df[df["is_spain"] == 1].copy()
    print("    Filtering complete.")

    # --- Step 4 & 5: Feature Engineering ---
    print("➡️ Step 4-5/7: Cleaning and engineering features...")
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    # Apply the function directly to the DataFrame
    popdens_cols = [c for c in df.columns if c.startswith("popdens_")]
    df["popdens"] = df.apply(assign_popdens, axis=1)
    df = df.drop(columns=popdens_cols)

    df["CLC_forest"] = df.apply(assign_clc, axis=1)
    print("    Feature engineering complete.")

    # --- Step 6: Final Preprocessing ---
    print("➡️ Step 6/7: One-hot encoding...")
    # Use pandas get_dummies for one-hot encoding
    df = pd.get_dummies(df, columns=["AutonomousCommunities", "month"], dtype=int)
    print("    One-hot encoding complete. Data is ready in RAM.")

    # --- Step 7: Create Datasets and Export ---
    print("➡️ Step 7/7: Creating, balancing, and exporting final datasets...")

    # --- Training Dataset (2008-2023) ---
    train_df = df[(df["year"] >= 2008) & (df["year"] <= 2023)]
    fires = train_df[train_df["is_fire"] == 1]
    nonfires = train_df[train_df["is_fire"] == 0]

    # Get counts directly with len()
    n_fires = len(fires)
    n_nonfires = len(nonfires)

    print(
        f"\nDEBUG CHECK (Train): Found {n_fires} fire samples and {n_nonfires} non-fire samples.\n"
    )

    if n_fires > 0 and n_nonfires > 0:
        sampling_fraction = n_fires / n_nonfires
        nonfires_sampled = nonfires.sample(frac=sampling_fraction, random_state=42)
        train_balanced = pd.concat([fires, nonfires_sampled])
        print(
            f"    Training set: {n_fires} fire samples and {len(nonfires_sampled)} non-fire samples."
        )

        print("    Writing train.csv to disk...")
        # Pandas to_csv executes immediately
        train_balanced.to_csv("train.csv", index=False)
        print("    train.csv has been saved.")
    else:
        print(
            "\n    Skipping training set export: No fire or non-fire data in the date range."
        )

    # --- Test Dataset (2024) ---
    test_df = df[df["year"] == 2024]
    fires_2024 = test_df[test_df["is_fire"] == 1]
    nonfires_2024 = test_df[test_df["is_fire"] == 0]

    n_fires_2024 = len(fires_2024)
    n_nonfires_2024 = len(nonfires_2024)

    if n_fires_2024 > 0 and n_nonfires_2024 > 0:
        sampling_fraction_2024 = n_fires_2024 / n_nonfires_2024
        nonfires_2024_sampled = nonfires_2024.sample(
            frac=sampling_fraction_2024, random_state=42
        )
        test_balanced = pd.concat([fires_2024, nonfires_2024_sampled])
        print(
            f"\n    Test set: {n_fires_2024} fire samples and {len(nonfires_2024_sampled)} non-fire samples."
        )

        print("    Writing test.csv to disk...")
        test_balanced.to_csv("test.csv", index=False)
        print("    test.csv has been saved.")
    else:
        print(
            "\n    Skipping test set export: No fire or non-fire data in the date range."
        )

    end_time = time.time()
    print(f"\n🎉 Script finished successfully in {end_time - start_time:.2f} seconds.")
