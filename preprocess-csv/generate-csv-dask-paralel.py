import dask
import dask.dataframe as dd
import xarray as xr
import numpy as np
import time

# Imports for Dask client
from dask.distributed import Client, LocalCluster


# Helper functions can be defined at the top level
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


# Main execution block to ensure multiprocessing safety
if __name__ == "__main__":
    print("⚙️ Setting up optimized Dask LocalCluster for a high-performance machine...")
    cluster = LocalCluster(n_workers=10, threads_per_worker=2, memory_limit="5GB")
    client = Client(cluster)
    print(f"    Dask dashboard available at: {client.dashboard_link}")

    start_time = time.time()

    # --- Step 1: Load Dataset with Dask-friendly Chunks ---
    print("➡️ Step 1/7: Loading IberFire.nc dataset with larger chunks...")
    ds = xr.open_dataset("IberFire.nc", chunks="auto").unify_chunks()
    print("    Dataset loaded and re-chunked.")

    # --- Step 2: Convert to Dask DataFrame ---
    print("➡️ Step 2/7: Converting Xarray Dataset to Dask DataFrame...")
    ddf = ds.to_dask_dataframe()
    ddf = ddf.reset_index()
    print("    Conversion complete.")

    # --- Step 3: Filter Geospatial Area (on the DataFrame) ---
    print("➡️ Step 3/7: Filtering for mainland Spain...")
    # This filtering is now done efficiently on the Dask DataFrame
    ddf = ddf[ddf["is_spain"] == 1].copy()
    print("    Filtering complete.")

    # --- Step 4 & 5: Feature Engineering ---
    print("➡️ Step 4-5/7: Cleaning and engineering features...")
    ddf["year"] = ddf["time"].dt.year
    ddf["month"] = ddf["time"].dt.month
    # Keep this to drop x and y columns
    # ddf = ddf.drop(columns=["x", "y"])

    popdens_cols = [c for c in ddf.columns if c.startswith("popdens_")]
    ddf["popdens"] = ddf.map_partitions(
        lambda df: df.apply(assign_popdens, axis=1), meta=("popdens", "f8")
    )
    ddf = ddf.drop(columns=popdens_cols)

    ddf["CLC_forest"] = ddf.map_partitions(
        lambda df: df.apply(assign_clc, axis=1), meta=("CLC_forest", "f8")
    )
    print("    Feature engineering complete.")

    # --- Step 6: Final Preprocessing & Persisting to RAM ---
    print("➡️ Step 6/7: One-hot encoding and persisting to RAM...")
    ddf = ddf.categorize(columns=["AutonomousCommunities", "month"])
    ddf = dd.get_dummies(ddf, columns=["AutonomousCommunities", "month"])

    print("    Persisting cleaned data into memory...")
    ddf = ddf.persist()
    print("    Data persisted to RAM. Final computations will be much faster.")

    # --- Step 7: Create Datasets and Export ---
    print("➡️ Step 7/7: Creating, balancing, and exporting final datasets...")

    # --- Training Dataset (2008-2023) ---
    train_ddf = ddf[(ddf["year"] >= 2008) & (ddf["year"] <= 2023)]
    fires = train_ddf[train_ddf["is_fire"] == 1]
    nonfires = train_ddf[train_ddf["is_fire"] == 0]
    print(f"\nDEBUG CHECK About to dask.compute!.\n")
    # n_fires, n_nonfires = dask.compute(len(fires), len(nonfires))
    n_fires, n_nonfires = dask.compute(
        fires.map_partitions(len).sum(), nonfires.map_partitions(len).sum()
    )

    print(
        f"\nDEBUG CHECK (Train): Found {n_fires} fire samples and {n_nonfires} non-fire samples.\n"
    )

    if n_fires > 0 and n_nonfires > 0:
        sampling_fraction = n_fires / n_nonfires
        nonfires_sampled = nonfires.sample(frac=sampling_fraction, random_state=42)
        train_balanced = dd.concat([fires, nonfires_sampled])
        print(
            f"\n    Training set: {n_fires} fire samples and ~{int(n_nonfires * sampling_fraction)} non-fire samples."
        )
        print("    Writing train.csv to disk...")
        # Add compute=True to execute the write operation
        train_balanced.to_csv("train.csv", single_file=True, index=False, compute=True)
        print("    train.csv has been saved.")
    else:
        print(
            "\n    Skipping training set export: No fire or non-fire data in the date range."
        )

    # --- Test Dataset (2024) ---
    test_ddf = ddf[ddf["year"] == 2024]
    fires_2024 = test_ddf[test_ddf["is_fire"] == 1]
    nonfires_2024 = test_ddf[test_ddf["is_fire"] == 0]
    # n_fires_2024, n_nonfires_2024 = dask.compute(len(fires_2024), len(nonfires_2024))
    # FIXED: Use map_partitions(len).sum() instead of len()
    n_fires_2024, n_nonfires_2024 = dask.compute(
        fires_2024.map_partitions(len).sum(), nonfires_2024.map_partitions(len).sum()
    )

    if n_fires_2024 > 0 and n_nonfires_2024 > 0:
        sampling_fraction_2024 = n_fires_2024 / n_nonfires_2024
        nonfires_2024_sampled = nonfires_2024.sample(
            frac=sampling_fraction_2024, random_state=42
        )
        test_balanced = dd.concat([fires_2024, nonfires_2024_sampled])
        print(
            f"\n    Test set: {n_fires_2024} fire samples and ~{int(n_nonfires_2024 * sampling_fraction_2024)} non-fire samples."
        )
        print("    Writing test.csv to disk...")
        # Add compute=True to execute the write operation
        test_balanced.to_csv("test.csv", single_file=True, index=False, compute=True)
        print("    test.csv has been saved.")
    else:
        print(
            "\n    Skipping test set export: No fire or non-fire data in the date range."
        )

    end_time = time.time()
    print(f"\n🎉 Script finished successfully in {end_time - start_time:.2f} seconds.")
