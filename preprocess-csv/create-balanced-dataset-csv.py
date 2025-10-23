import xarray as xr
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

# --- Configuration ---
# Replace with the correct path to your .nc file
IBERFIRE_PATH = "../IberFire/IberFire.nc"
OUTPUT_CSV_PATH = "iberfire_preprocessed_balanced.csv"
RANDOM_STATE = 42  # For reproducibility of the sampling

# Configure logging to track the process
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_year_specific_features(year):
    """
    Returns the names of the CLC and popdens columns for a specific year.
    This helps select the correct version of the data to prevent data leakage.
    """
    # [cite_start]Define the CLC update years [cite: 7]
    if year < 2012:
        clc_year = 2006
    elif year < 2018:
        clc_year = 2012
    else:
        clc_year = 2018

    # [cite_start]Population density is available annually from 2008 to 2020 [cite: 8]
    popdens_year = min(max(year, 2008), 2020)

    return f"popdens_{popdens_year}", [f"CLC_{clc_year}_{i}" for i in range(1, 45)]


def get_balanced_indices(ds):
    """
    Identifies the indices for a balanced dataset in a memory-efficient,
    iterative manner, avoiding loading massive data chunks at once.
    """
    logging.info(
        "Identifying indices for the balanced dataset (iterative memory-safe mode)..."
    )

    # Load the Spain mask, which is small and fits in memory.
    is_spain_mask = ds["is_spain"].astype(bool).compute()
    # Keep the fire mask as a lazy dask array for efficient reads.
    lazy_fire_mask = ds["is_fire"].astype(bool)

    # --- 1. Get fire indices iteratively (day by day) ---
    logging.info("Finding fire records (daily scan)...")
    fire_records = []

    # Iterate through each timestamp in the dataset.
    for t in tqdm(ds["time"].values, desc="Scanning for fires"):
        # Load the 2D slice for a single day (very lightweight operation).
        daily_fire_slice = lazy_fire_mask.sel(time=t).values

        # Apply the Spain mask to only consider the territory.
        fires_in_spain_today = daily_fire_slice & is_spain_mask

        # If any fires occurred on this day, find their coordinates.
        if fires_in_spain_today.any():
            y_indices, x_indices = np.where(fires_in_spain_today)

            # Add the found records to our list.
            for y_idx, x_idx in zip(y_indices, x_indices):
                fire_records.append(
                    {"time": t, "y": ds.y.values[y_idx], "x": ds.x.values[x_idx]}
                )

    fire_indices_df = pd.DataFrame(fire_records)
    logging.info(f"Found {len(fire_indices_df)} fire records.")

    # --- 2. Sample non-fire indices (already safe logic) ---
    logging.info("Sampling non-fire records...")
    n_samples = len(fire_indices_df)
    non_fire_samples = []

    y_coords_idx, x_coords_idx = np.where(is_spain_mask)
    valid_land_indices = list(zip(y_coords_idx, x_coords_idx))

    time_coords = ds["time"].values
    rng = np.random.default_rng(RANDOM_STATE)

    pbar = tqdm(total=n_samples, desc="Sampling non-fire instances")
    while len(non_fire_samples) < n_samples:
        rand_time_idx = rng.integers(0, len(time_coords))
        rand_land_idx = rng.integers(0, len(valid_land_indices))

        y_idx, x_idx = valid_land_indices[rand_land_idx]

        # .compute() gets the actual value from the lazy dask array.
        is_fire_value = lazy_fire_mask[rand_time_idx, y_idx, x_idx].compute()

        if not is_fire_value:
            sample = {
                "time": time_coords[rand_time_idx],
                "y": ds.y.values[y_idx],
                "x": ds.x.values[x_idx],
            }
            if sample not in non_fire_samples:
                non_fire_samples.append(sample)
                pbar.update(1)
    pbar.close()

    non_fire_indices_df = pd.DataFrame(non_fire_samples)

    # --- 3. Combine the indices ---
    balanced_indices_df = pd.concat(
        [fire_indices_df, non_fire_indices_df], ignore_index=True
    )
    logging.info(f"Total indices in the balanced dataset: {len(balanced_indices_df)}")

    return balanced_indices_df


def process_data(ds, indices_df):
    """
    Extracts and processes data only for the selected indices to save memory.
    """
    processed_records = []

    logging.info("Starting selective data extraction and processing...")
    # Group by time to optimize file reading
    for timestamp, group in tqdm(
        indices_df.groupby("time"), desc="Processing day by day"
    ):
        date = pd.to_datetime(timestamp)
        year = date.year

        # Select data for the (y, x) points at this specific timestamp
        points = ds.sel(
            time=timestamp, y=xr.DataArray(group["y"]), x=xr.DataArray(group["x"])
        )

        # Load only this small subset into memory
        points_df = points.to_dataframe().reset_index()

        # Logic to prevent data leakage with popdens and CLC
        popdens_col, clc_cols_year = get_year_specific_features(year)

        # Rename columns to a generic name
        points_df["popdens"] = points_df[popdens_col]

        clc_rename_map = {
            old_col: f"CLC_proportion_{i}" for i, old_col in enumerate(clc_cols_year, 1)
        }
        points_df.rename(columns=clc_rename_map, inplace=True)

        # Add month feature for later one-hot encoding
        points_df["month"] = date.month

        processed_records.append(points_df)

    logging.info("Assembling the final DataFrame...")
    final_df = pd.concat(processed_records, ignore_index=True)
    return final_df


def final_cleanup_and_encoding(df):
    """
    Performs final cleanup, one-hot encoding, and feature selection.
    """
    logging.info("Performing final cleanup and encoding...")

    # [cite_start]1. One-hot encode categorical features [cite: 832, 834]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    # AutonomousCommunities
    ac_encoded = ohe.fit_transform(df[["AutonomousCommunities"]])
    ac_df = pd.DataFrame(
        ac_encoded, columns=[f"AC_{cat}" for cat in ohe.categories_[0]]
    )

    # Month
    month_encoded = ohe.fit_transform(df[["month"]])
    month_df = pd.DataFrame(
        month_encoded, columns=[f"month_{int(cat)}" for cat in ohe.categories_[0]]
    )

    df = pd.concat([df.reset_index(drop=True), ac_df, month_df], axis=1)

    # 2. Remove unnecessary columns
    # [cite_start]Exclude auxiliary features [cite: 826]
    cols_to_drop = ["x_index", "y_index"]

    # Exclude original categorical columns
    cols_to_drop.extend(["AutonomousCommunities", "month"])

    # Exclude all year-specific popdens and CLC columns
    all_popdens_cols = [f"popdens_{y}" for y in range(2008, 2021)]
    all_clc_cols = [f"CLC_{y}_{i}" for y in [2006, 2012, 2018] for i in range(1, 45)]
    cols_to_drop.extend(all_popdens_cols)
    cols_to_drop.extend(all_clc_cols)

    # Remove columns that exist in the DataFrame
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    return df


def main():
    """
    Orchestrates the entire preprocessing pipeline.
    """
    # Using 'with' ensures the file is closed properly
    try:
        with xr.open_dataset(IBERFIRE_PATH, chunks="auto") as ds:
            logging.info("IberFire dataset loaded successfully (lazy mode).")

            # Step 1: Get the indices for a balanced dataset
            indices_df = get_balanced_indices(ds)

            # Step 2: Process the data for only those indices
            processed_df = process_data(ds, indices_df)

    except FileNotFoundError:
        logging.error(f"Error: File not found at {IBERFIRE_PATH}")
        return

    # Step 3: Final cleanup and encoding
    final_df = final_cleanup_and_encoding(processed_df)

    # Step 4: Save the result
    logging.info(f"Saving processed dataset to {OUTPUT_CSV_PATH}")
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    logging.info("Process completed successfully!")
    print("\nFinal DataFrame Summary:")
    print(final_df.info())


if __name__ == "__main__":
    main()
