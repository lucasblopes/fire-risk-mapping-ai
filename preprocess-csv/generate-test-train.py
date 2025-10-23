import pandas as pd
import xarray as xr
import numpy as np
import time
from pathlib import Path
import os
import gc


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


def process_single_timestep(ds, time_idx):
    """Process a SINGLE time step to minimize memory usage"""

    # Select just one time step
    ds_single = ds.isel(time=time_idx)

    # Convert to dataframe - much smaller now (just one time slice)
    df = ds_single.to_dataframe().reset_index()

    # Filter for mainland Spain immediately
    df = df[df["is_spain"] == 1].copy()

    if len(df) == 0:
        return pd.DataFrame()

    # Feature engineering
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    # Population density assignment
    popdens_cols = [c for c in df.columns if c.startswith("popdens_")]
    df["popdens"] = df.apply(assign_popdens, axis=1)
    df = df.drop(columns=popdens_cols)

    # CLC forest assignment
    df["CLC_forest"] = df.apply(assign_clc, axis=1)

    # One-hot encode immediately
    df = pd.get_dummies(df, columns=["AutonomousCommunities", "month"], dtype=int)

    return df


def collect_and_save_fires(ds, year_start, year_end, output_file):
    """Collect fire samples ONE TIMESTEP AT A TIME"""

    print(f"  Collecting fire samples from {year_start} to {year_end}...")

    total_time_steps = ds.time.size
    fire_count = 0
    first_chunk = True

    for time_idx in range(total_time_steps):
        # Check year first to skip irrelevant timesteps
        current_time = pd.Timestamp(ds.time.values[time_idx])
        current_year = current_time.year

        if current_year < year_start or current_year > year_end:
            continue

        if time_idx % 100 == 0:
            print(
                f"    Processing timestep {time_idx}/{total_time_steps} ({current_time.date()})..."
            )

        # Process single timestep
        df = process_single_timestep(ds, time_idx)

        if len(df) == 0:
            continue

        # Filter for fire events only
        fire_df = df[df["is_fire"] == 1]

        if len(fire_df) > 0:
            # Write to CSV
            fire_df.to_csv(
                output_file, mode="a", header=first_chunk, index=False, sep=";"
            )
            fire_count += len(fire_df)
            first_chunk = False

            if fire_count % 1000 == 0:
                print(f"      Collected {fire_count} fire samples so far...")

        # Explicit cleanup
        del df
        if "fire_df" in locals():
            del fire_df

        # Force garbage collection every 50 timesteps
        if time_idx % 50 == 0:
            gc.collect()

    print(f"  Total fire samples saved: {fire_count}")
    return fire_count


def collect_and_save_nonfires(
    ds, year_start, year_end, n_samples, output_file, random_seed=42
):
    """Collect non-fire samples ONE TIMESTEP AT A TIME"""

    print(
        f"  Collecting {n_samples} non-fire samples from {year_start} to {year_end}..."
    )

    np.random.seed(random_seed)
    samples_collected = 0
    total_time_steps = ds.time.size
    first_chunk = True

    # Calculate how many samples to collect per timestep (rough estimate)
    # Assuming ~1000 valid cells per timestep, we need to sample aggressively
    samples_per_timestep = max(1, n_samples // 100)

    for time_idx in range(total_time_steps):
        if samples_collected >= n_samples:
            print(f"  Target reached: {samples_collected}/{n_samples}")
            break

        # Check year first
        current_time = pd.Timestamp(ds.time.values[time_idx])
        current_year = current_time.year

        if current_year < year_start or current_year > year_end:
            continue

        if time_idx % 100 == 0:
            print(
                f"    Processing timestep {time_idx}/{total_time_steps} ({current_time.date()})..."
            )
            print(f"      Collected {samples_collected}/{n_samples} samples so far")

        # Process single timestep
        df = process_single_timestep(ds, time_idx)

        if len(df) == 0:
            continue

        # Filter for non-fire events only
        nonfire_df = df[df["is_fire"] == 0]

        if len(nonfire_df) > 0:
            # Sample from this timestep
            n_to_sample = min(
                samples_per_timestep, len(nonfire_df), n_samples - samples_collected
            )

            if n_to_sample > 0:
                sampled = nonfire_df.sample(
                    n=n_to_sample, random_state=random_seed + time_idx
                )

                # Write to CSV
                sampled.to_csv(
                    output_file, mode="a", header=first_chunk, index=False, sep=";"
                )
                samples_collected += len(sampled)
                first_chunk = False

        # Explicit cleanup
        del df
        if "nonfire_df" in locals():
            del nonfire_df
        if "sampled" in locals():
            del sampled

        # Force garbage collection every 50 timesteps
        if time_idx % 50 == 0:
            gc.collect()

    print(f"  Total non-fire samples saved: {samples_collected}")
    return samples_collected


def shuffle_csv_file(input_file, output_file, random_seed=42, chunksize=50000):
    """Shuffle a CSV file - if too large, use chunked approach"""

    print(f"  Shuffling {input_file}...")

    # Try to load entire file
    try:
        df = pd.read_csv(input_file, sep=";")
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        df.to_csv(output_file, index=False, sep=";")
        print(f"  Shuffled dataset saved to {output_file}")
    except MemoryError:
        print(f"  File too large to shuffle in memory, copying without shuffle...")
        # Just copy the file if shuffling would exceed memory
        import shutil

        shutil.copy(input_file, output_file)
        print(
            f"  Dataset copied to {output_file} (not shuffled due to memory constraints)"
        )


if __name__ == "__main__":
    print("Starting IberFire balanced dataset generation...")
    print("=" * 60)
    print("Processing ONE TIMESTEP AT A TIME to minimize memory usage")
    print("=" * 60)

    start_time = time.time()

    # Load dataset metadata
    print("\nStep 1: Loading IberFire.nc metadata...")
    ds = xr.open_dataset("IberFire.nc")
    print(f"  Dataset dimensions: {ds.dims}")
    print(f"  Total time steps: {ds.time.size}")
    print(f"  Date range: {ds.time.values[0]} to {ds.time.values[-1]}")

    # --- TRAINING DATASET (2008-2023) ---
    print("\n" + "=" * 60)
    print("TRAINING DATASET (2008-2023)")
    print("=" * 60)

    # Temporary files
    temp_train_fires = "temp_train_fires.csv"
    temp_train_nonfires = "temp_train_nonfires.csv"
    temp_train_combined = "temp_train_combined.csv"

    # Remove temp files if they exist
    for f in [temp_train_fires, temp_train_nonfires, temp_train_combined]:
        if Path(f).exists():
            os.remove(f)

    # Collect fire samples
    n_train_fires = collect_and_save_fires(ds, 2008, 2023, temp_train_fires)

    if n_train_fires == 0:
        print("ERROR: No fire samples found for training period!")
        exit(1)

    # Force garbage collection
    gc.collect()

    # Collect non-fire samples
    n_train_nonfires = collect_and_save_nonfires(
        ds, 2008, 2023, n_train_fires, temp_train_nonfires, random_seed=42
    )

    if n_train_nonfires == 0:
        print("ERROR: No non-fire samples found for training period!")
        exit(1)

    print(f"\n  Training set composition:")
    print(f"    Fire samples: {n_train_fires}")
    print(f"    Non-fire samples: {n_train_nonfires}")
    print(f"    Total samples: {n_train_fires + n_train_nonfires}")

    # Combine files
    print("\n  Combining train files...")
    fires_df = pd.read_csv(temp_train_fires, sep=";")
    nonfires_df = pd.read_csv(temp_train_nonfires, sep=";")
    combined_df = pd.concat([fires_df, nonfires_df], ignore_index=True)
    combined_df.to_csv(temp_train_combined, index=False, sep=";")
    del fires_df, nonfires_df, combined_df
    gc.collect()

    # Shuffle and save final
    shuffle_csv_file(temp_train_combined, "iberfire_train.csv", random_seed=42)

    # Clean up temp files
    for f in [temp_train_fires, temp_train_nonfires, temp_train_combined]:
        if Path(f).exists():
            os.remove(f)

    print("  Training dataset saved successfully!")

    # --- TEST DATASET (2024) ---
    print("\n" + "=" * 60)
    print("TEST DATASET (2024)")
    print("=" * 60)

    # Temporary files
    temp_test_fires = "temp_test_fires.csv"
    temp_test_nonfires = "temp_test_nonfires.csv"
    temp_test_combined = "temp_test_combined.csv"

    # Remove temp files if they exist
    for f in [temp_test_fires, temp_test_nonfires, temp_test_combined]:
        if Path(f).exists():
            os.remove(f)

    # Collect fire samples
    n_test_fires = collect_and_save_fires(ds, 2024, 2024, temp_test_fires)

    if n_test_fires == 0:
        print("WARNING: No fire samples found for 2024!")
    else:
        gc.collect()

        # Collect non-fire samples
        n_test_nonfires = collect_and_save_nonfires(
            ds, 2024, 2024, n_test_fires, temp_test_nonfires, random_seed=42
        )

        if n_test_nonfires == 0:
            print("WARNING: No non-fire samples found for 2024!")
        else:
            print(f"\n  Test set composition:")
            print(f"    Fire samples: {n_test_fires}")
            print(f"    Non-fire samples: {n_test_nonfires}")
            print(f"    Total samples: {n_test_fires + n_test_nonfires}")

            # Combine files
            print("\n  Combining test files...")
            fires_df = pd.read_csv(temp_test_fires, sep=";")
            nonfires_df = pd.read_csv(temp_test_nonfires, sep=";")
            combined_df = pd.concat([fires_df, nonfires_df], ignore_index=True)
            combined_df.to_csv(temp_test_combined, index=False, sep=";")
            del fires_df, nonfires_df, combined_df
            gc.collect()

            # Shuffle and save final
            shuffle_csv_file(temp_test_combined, "iberfire_test.csv", random_seed=42)

            # Clean up temp files
            for f in [temp_test_fires, temp_test_nonfires, temp_test_combined]:
                if Path(f).exists():
                    os.remove(f)

            print("  Test dataset saved successfully!")

    end_time = time.time()

    print("\n" + "=" * 60)
    print(f"Dataset generation completed in {end_time - start_time:.2f} seconds!")
    print("=" * 60)
    print("\nGenerated files:")
    if Path("iberfire_train.csv").exists():
        print(
            f"  iberfire_train.csv ({Path('iberfire_train.csv').stat().st_size / 1024**2:.1f} MB)"
        )
    if Path("iberfire_test.csv").exists():
        print(
            f"  iberfire_test.csv ({Path('iberfire_test.csv').stat().st_size / 1024**2:.1f} MB)"
        )
