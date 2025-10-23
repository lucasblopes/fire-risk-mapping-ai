import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import gc
import os

# --- CONFIGURATION ---
IBERFIRE_NC_PATH = "IberFire.nc"
TRAIN_OUTPUT_CSV_PATH = "iberfire_training_balanced_dataset.csv"
TEST_OUTPUT_CSV_PATH = "iberfire_testing_dataset.csv"
TRAIN_START_DATE = "2008-01-01"
TRAIN_END_DATE = "2023-12-31"
TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2024-12-31"
RANDOM_SEED = 42
PROCESS_CHUNK_SIZE = 20000


def get_clc_column_mappings():
    """Defines the mappings from year-specific CLC columns to generic names."""
    clc_level3_suffixes = [f"_{i}" for i in range(1, 45)]
    clc_level2_suffixes = [
        "_urban_fabric_proportion",
        "_industrial_proportion",
        "_mine_proportion",
        "_artificial_vegetation_proportion",
        "_arable_land_proportion",
        "_permanent_crops_proportion",
        "_heterogeneous_agriculture_proportion",
        "_forest_proportion",
        "_scrub_proportion",
        "_open_space_proportion",
        "_inland_wetlands_proportion",
        "_maritime_wetlands_proportion",
        "_inland_waters_proportion",
        "_marine_waters_proportion",
    ]
    clc_level1_suffixes = [
        "_artificial_proportion",
        "_agricultural_proportion",
        "_forest_and_semi_natural_proportion",
        "_wetlands_proportion",
        "_waterbody_proportion",
    ]
    all_suffixes = clc_level3_suffixes + clc_level2_suffixes + clc_level1_suffixes
    clc_mappings = {
        "generic": [f"CLC{suffix}" for suffix in all_suffixes],
        "2006": [f"CLC_2006{suffix}" for suffix in all_suffixes],
        "2012": [f"CLC_2012{suffix}" for suffix in all_suffixes],
        "2018": [f"CLC_2018{suffix}" for suffix in all_suffixes],
    }
    return clc_mappings


def get_final_column_order():
    """Defines the exact order and names of the 145 columns for the final CSV."""
    clc_mappings = get_clc_column_mappings()
    ac_cols = [f"AC_{i}" for i in range(17)]
    month_cols = [f"month_{i}" for i in range(1, 13)]
    base_features = [
        "is_fire",
        "x_coordinate",
        "y_coordinate",
        "is_natura2000",
        "elevation_mean",
        "elevation_stdev",
        "slope_mean",
        "slope_stdev",
        "roughness_mean",
        "roughness_stdev",
        "dist_to_roads_mean",
        "dist_to_roads_stdev",
        "dist_to_waterways_mean",
        "dist_to_waterways_stdev",
        "dist_to_railways_mean",
        "dist_to_railways_stdev",
        "is_holiday",
        "t2m_mean",
        "t2m_max",
        "t2m_min",
        "t2m_range",
        "RH_mean",
        "RH_max",
        "RH_min",
        "RH_range",
        "surface_pressure_mean",
        "surface_pressure_max",
        "surface_pressure_min",
        "surface_pressure_range",
        "total_precipitation_mean",
        "wind_speed_mean",
        "wind_speed_max",
        "wind_direction_mean",
        "wind_direction_at_max_speed",
        "FAPAR",
        "LAI",
        "NDVI",
        "LST",
        "SWI_001",
        "SWI_005",
        "SWI_010",
        "SWI_020",
        "popdens",
    ]
    aspect_features = [f"aspect_{i}" for i in range(1, 9)] + ["aspect_NODATA"]
    final_cols = (
        ["time"]
        + base_features
        + aspect_features
        + clc_mappings["generic"]
        + ac_cols
        + month_cols
    )
    if len(final_cols) != 145:
        print(f"Warning: Expected 145 columns, but generated {len(final_cols)}.")
    return final_cols


def create_balanced_dataset(
    ds, clc_mappings, final_cols, output_path, start_date, end_date
):
    """
    Main function to perform sampling, data extraction, and feature engineering
    for a specified time range, writing the result to a CSV file.
    """
    if os.path.exists(output_path):
        print(f"File '{output_path}' already exists. Skipping creation.")
        return

    rng = np.random.default_rng(RANDOM_SEED)
    print(f"\nProcessing data for {start_date} to {end_date}...")
    ds_filtered = ds.sel(time=slice(start_date, end_date))
    is_spain_mask = (ds["is_spain"] == 1).compute()
    ds_filtered = ds_filtered.where(is_spain_mask, drop=True)

    print("Identifying all fire locations...")
    fire_mask = ds_filtered["is_fire"] == 1
    fire_indices = np.argwhere(fire_mask.compute().values)
    time_coords = ds_filtered.time.values
    fire_coords = [(time_coords[i[0]], i[1], i[2]) for i in fire_indices]
    n_fire = len(fire_coords)
    print(f"Found {n_fire} fire instances.")

    if n_fire == 0:
        print(
            f"Warning: No fire instances found between {start_date} and {end_date}. Cannot create a balanced dataset."
        )
        return

    print("Sampling non-fire locations...")
    non_fire_coords = []
    non_fire_mask = (ds_filtered["is_fire"] == 0) & (ds_filtered["is_near_fire"] == 0)
    total_non_fire_candidates = non_fire_mask.sum().compute().item()

    if total_non_fire_candidates == 0:
        print("Warning: No non-fire candidates found for this period.")
        sampling_fraction = 0
    else:
        sampling_fraction = min(n_fire / total_non_fire_candidates, 1)

    print(
        f"Sampling ~{n_fire} non-fire instances with a fraction of {sampling_fraction:.6f}"
    )

    for year_val, ds_year in tqdm(
        ds_filtered.groupby("time.year"), desc="Sampling non-fire by year"
    ):
        non_fire_mask_year = (ds_year["is_fire"] == 0) & (ds_year["is_near_fire"] == 0)
        year_indices = np.argwhere(non_fire_mask_year.compute().values)
        if year_indices.shape[0] == 0:
            continue
        n_to_sample = int(np.round(len(year_indices) * sampling_fraction))
        if n_to_sample > 0:
            sampled_indices = rng.choice(
                year_indices, size=n_to_sample, replace=False, axis=0
            )
            year_time_coords = ds_year.time.values
            for idx in sampled_indices:
                non_fire_coords.append((year_time_coords[idx[0]], idx[1], idx[2]))

    print(f"Sampled {len(non_fire_coords)} non-fire instances.")
    all_coords = fire_coords + non_fire_coords
    rng.shuffle(all_coords)
    total_samples = len(all_coords)
    print(f"Total balanced dataset size: {total_samples} instances.")

    del ds_filtered, fire_mask, fire_indices, non_fire_mask
    gc.collect()

    print("\nProcessing data in chunks and writing to CSV...")

    # Get a list of all variables to select from the dataset
    all_vars = list(ds.data_vars.keys())

    # Write header
    pd.DataFrame(columns=final_cols).to_csv(output_path, index=False)

    for i in tqdm(
        range(0, total_samples, PROCESS_CHUNK_SIZE), desc="Processing Chunks"
    ):
        chunk_coords = all_coords[i : i + PROCESS_CHUNK_SIZE]
        time_vals, y_indices, x_indices = zip(*chunk_coords)
        time_da = xr.DataArray(list(time_vals), dims="points")
        y_da = xr.DataArray(list(y_indices), dims="points")
        x_da = xr.DataArray(list(x_indices), dims="points")

        chunk_ds = ds[all_vars].sel(time=time_da).isel(y=y_da, x=x_da)
        df_chunk = chunk_ds.to_dataframe().reset_index()

        # --- Feature Engineering ---
        df_chunk["year"] = df_chunk["time"].dt.year
        df_chunk["month"] = df_chunk["time"].dt.month

        # a) Map population density
        df_chunk["popdens"] = np.nan
        for year, group in df_chunk.groupby("year"):
            popdens_col = f"popdens_{year}"
            if popdens_col in df_chunk.columns:
                df_chunk.loc[group.index, "popdens"] = group[popdens_col]

        # b) Map CLC features
        for generic_col, clc_2006_col, clc_2012_col, clc_2018_col in zip(
            clc_mappings["generic"],
            clc_mappings["2006"],
            clc_mappings["2012"],
            clc_mappings["2018"],
        ):
            df_chunk[generic_col] = np.nan
            idx_2006 = df_chunk["year"] < 2012
            df_chunk.loc[idx_2006, generic_col] = df_chunk.loc[
                idx_2006, clc_2006_col
            ].values
            idx_2012 = (df_chunk["year"] >= 2012) & (df_chunk["year"] < 2018)
            df_chunk.loc[idx_2012, generic_col] = df_chunk.loc[
                idx_2012, clc_2012_col
            ].values
            idx_2018 = df_chunk["year"] >= 2018
            df_chunk.loc[idx_2018, generic_col] = df_chunk.loc[
                idx_2018, clc_2018_col
            ].values

        # c) One-Hot Encoding
        df_chunk["AutonomousCommunities"] = df_chunk["AutonomousCommunities"].astype(
            int
        )
        month_dummies = pd.get_dummies(df_chunk["month"], prefix="month", dtype=int)
        for m in range(1, 13):
            if f"month_{m}" not in month_dummies.columns:
                month_dummies[f"month_{m}"] = 0
        ac_dummies = pd.get_dummies(
            df_chunk["AutonomousCommunities"], prefix="AC", dtype=int
        )
        for ac in range(17):
            if f"AC_{ac}" not in ac_dummies.columns:
                ac_dummies[f"AC_{ac}"] = 0
        df_chunk = pd.concat([df_chunk, month_dummies, ac_dummies], axis=1)

        # Drop temporary columns before final selection
        cols_to_drop = [
            "popdens_2010",
            "popdens_2015",
            "popdens_2020",
            "is_near_fire",
            "is_spain",
            "x_index",
            "y_index",
            "CLC_2006_urban_fabric_proportion",
            "CLC_2012_urban_fabric_proportion",
            "CLC_2018_urban_fabric_proportion",
        ]
        df_chunk = df_chunk.drop(
            columns=[c for c in cols_to_drop if c in df_chunk.columns]
        )

        # Fill missing values within the chunk
        for col in df_chunk.columns:
            if df_chunk[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_chunk[col]):
                    col_avg = df_chunk[col].mean()
                    df_chunk[col] = df_chunk[col].fillna(col_avg)
                else:
                    df_chunk[col] = df_chunk[col].fillna("missing")

        # Finalize and Append
        df_final_chunk = df_chunk[final_cols]

        df_final_chunk.to_csv(output_path, mode="a", header=False, index=False)
        del df_chunk, chunk_ds, df_final_chunk, time_da, y_da, x_da
        gc.collect()

    print(f"Processing complete. Balanced dataset saved to '{output_path}'")


if __name__ == "__main__":
    try:
        print(f"Loading dataset '{IBERFIRE_NC_PATH}' with Dask chunks...")
        ds = xr.open_dataset(IBERFIRE_NC_PATH, chunks="auto")
        ds = ds.chunk({"time": 120})
    except FileNotFoundError:
        sys.exit(
            f"Error: The file '{IBERFIRE_NC_PATH}' was not found. Please check the path."
        )

    clc_mappings = get_clc_column_mappings()
    final_cols = get_final_column_order()

    # Create the training dataset
    create_balanced_dataset(
        ds,
        clc_mappings,
        final_cols,
        TRAIN_OUTPUT_CSV_PATH,
        TRAIN_START_DATE,
        TRAIN_END_DATE,
    )

    # Create the testing dataset
    create_balanced_dataset(
        ds,
        clc_mappings,
        final_cols,
        TEST_OUTPUT_CSV_PATH,
        TEST_START_DATE,
        TEST_END_DATE,
    )
