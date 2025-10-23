import pandas as pd
import xarray as xr
import numpy as np
import time
from pathlib import Path


# Helper functions remain the same
def assign_popdens(row):
    """Selects the correct population density column based on the row's year."""
    year = row["year"]
    col = f"popdens_{year}"
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


def process_time_chunk(ds, time_indices, chunk_size=100):
    """Process a subset of time indices efficiently"""

    # Select the time subset
    ds_subset = ds.isel(time=time_indices)

    # Convert to dataframe - much smaller now
    df_chunk = ds_subset.to_dataframe().reset_index()

    # Filter for mainland Spain immediately to reduce memory
    df_chunk = df_chunk[df_chunk["is_spain"] == 1].copy()

    if len(df_chunk) == 0:
        return pd.DataFrame()  # Return empty DataFrame if no Spanish data

    # Feature engineering
    df_chunk["year"] = df_chunk["time"].dt.year
    df_chunk["month"] = df_chunk["time"].dt.month

    # Population density assignment
    popdens_cols = [c for c in df_chunk.columns if c.startswith("popdens_")]
    df_chunk["popdens"] = df_chunk.apply(assign_popdens, axis=1)
    df_chunk = df_chunk.drop(columns=popdens_cols)

    # CLC forest assignment
    df_chunk["CLC_forest"] = df_chunk.apply(assign_clc, axis=1)

    return df_chunk


def get_categorical_columns_info(ds):
    """Get information about categorical columns for consistent one-hot encoding"""
    # We need to get all unique values for consistent encoding across chunks
    # Load a small sample to get the categorical info
    sample = ds.isel(time=slice(0, min(10, ds.time.size)))
    sample_df = sample.to_dataframe().reset_index()
    sample_df = sample_df[sample_df["is_spain"] == 1]

    if len(sample_df) == 0:
        return [], []

    # Get unique autonomous communities and months (1-12)
    autonomous_communities = sorted(
        sample_df["AutonomousCommunities"].dropna().unique()
    )
    months = list(range(1, 13))  # All months

    return autonomous_communities, months


if __name__ == "__main__":
    print("  Starting memory-optimized data processing...")

    start_time = time.time()

    # --- Step 1: Load Dataset Metadata ---
    print("  Step 1/8: Loading dataset metadata...")
    ds = xr.open_dataset("IberFire.nc")
    print(f"    Dataset shape: {ds.dims}")
    print(f"    Total time steps: {ds.time.size}")

    # --- Step 2: Get Categorical Info ---
    print("  Step 2/8: Analyzing categorical variables...")
    autonomous_communities, months = get_categorical_columns_info(ds)
    print(f"    Found {len(autonomous_communities)} autonomous communities")
    print(f"    Months: {months}")

    # --- Step 3: Process Data in Time Chunks ---
    print("  Step 3/8: Processing data in temporal chunks...")

    chunk_size = 50  # Process 50 time steps at once - adjust based on memory usage
    total_time_steps = ds.time.size
    processed_chunks = []

    for i in range(0, total_time_steps, chunk_size):
        end_idx = min(i + chunk_size, total_time_steps)
        time_indices = slice(i, end_idx)

        print(
            f"    Processing time steps {i} to {end_idx - 1} ({end_idx - i} steps)..."
        )

        # Process this chunk
        chunk_df = process_time_chunk(ds, time_indices)

        if len(chunk_df) > 0:
            processed_chunks.append(chunk_df)

        # Print memory usage info
        print(f"    Chunk processed. Current chunk size: {len(chunk_df)} rows")

    # --- Step 4: Combine All Chunks ---
    print("  Step 4/8: Combining all processed chunks...")
    if processed_chunks:
        df = pd.concat(processed_chunks, ignore_index=True)
        print(f"    Combined dataframe shape: {df.shape}")
    else:
        print("    No data found! Check your dataset.")
        exit(1)

    # Clear chunks from memory
    del processed_chunks

    # --- Step 5: One-Hot Encoding ---
    print("  Step 5/8: Performing one-hot encoding...")

    # Ensure all categorical values are present for consistent encoding
    df["AutonomousCommunities"] = df["AutonomousCommunities"].astype("category")
    df["month"] = df["month"].astype("category")

    # Set categories to ensure consistent encoding
    df["AutonomousCommunities"] = df["AutonomousCommunities"].cat.set_categories(
        autonomous_communities
    )
    df["month"] = df["month"].cat.set_categories(months)

    # One-hot encode
    df = pd.get_dummies(df, columns=["AutonomousCommunities", "month"], dtype=int)
    print(f"    One-hot encoding complete. Final shape: {df.shape}")

    # --- Step 6: Memory Optimization ---
    print("  Step 6/8: Optimizing data types...")

    # Optimize memory usage by downcasting numeric types where possible
    for col in df.select_dtypes(include=["int64"]).columns:
        if df[col].min() >= 0 and df[col].max() <= 255:
            df[col] = df[col].astype("uint8")
        elif df[col].min() >= -128 and df[col].max() <= 127:
            df[col] = df[col].astype("int8")
        elif df[col].min() >= 0 and df[col].max() <= 65535:
            df[col] = df[col].astype("uint16")
        elif df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype("int16")

    # Convert float64 to float32 where precision allows
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    print(f"    Memory optimization complete.")

    # --- Step 7: Create Training Dataset (2008-2023) ---
    print("  Step 7/8: Creating and balancing training dataset...")

    train_df = df[(df["year"] >= 2008) & (df["year"] <= 2023)].copy()
    fires = train_df[train_df["is_fire"] == 1]
    nonfires = train_df[train_df["is_fire"] == 0]

    n_fires = len(fires)
    n_nonfires = len(nonfires)

    print(f"    Training data: {n_fires} fire samples, {n_nonfires} non-fire samples")

    if n_fires > 0 and n_nonfires > 0:
        sampling_fraction = n_fires / n_nonfires
        print(f"    Sampling fraction: {sampling_fraction:.6f}")

        nonfires_sampled = nonfires.sample(frac=sampling_fraction, random_state=42)
        train_balanced = pd.concat([fires, nonfires_sampled], ignore_index=True)

        print(f"    Balanced training set: {len(train_balanced)} total samples")
        print("    Writing train.csv...")
        train_balanced.to_csv("train.csv", index=False)
        print("     train.csv saved successfully")

        # Clear training data from memory
        del train_df, fires, nonfires, nonfires_sampled, train_balanced
    else:
        print("      Skipping training set: insufficient data")

    # --- Step 8: Create Test Dataset (2024) ---
    print("  Step 8/8: Creating and balancing test dataset...")

    test_df = df[df["year"] == 2024].copy()
    fires_2024 = test_df[test_df["is_fire"] == 1]
    nonfires_2024 = test_df[test_df["is_fire"] == 0]

    n_fires_2024 = len(fires_2024)
    n_nonfires_2024 = len(nonfires_2024)

    print(
        f"    Test data: {n_fires_2024} fire samples, {n_nonfires_2024} non-fire samples"
    )

    if n_fires_2024 > 0 and n_nonfires_2024 > 0:
        sampling_fraction_2024 = n_fires_2024 / n_nonfires_2024
        print(f"    Sampling fraction: {sampling_fraction_2024:.6f}")

        nonfires_2024_sampled = nonfires_2024.sample(
            frac=sampling_fraction_2024, random_state=42
        )
        test_balanced = pd.concat(
            [fires_2024, nonfires_2024_sampled], ignore_index=True
        )

        print(f"    Balanced test set: {len(test_balanced)} total samples")
        print("    Writing test.csv...")
        test_balanced.to_csv("test.csv", index=False)
        print("     test.csv saved successfully")
    else:
        print("      Skipping test set: insufficient data")

    end_time = time.time()
    print(f"\n Processing completed in {end_time - start_time:.2f} seconds!")

    # Final memory info
    print(f"\n Final dataset info:")
    print(f"    Total processed rows: {len(df)}")
    print(f"    Total columns: {len(df.columns)}")
    print(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
