import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import timm
from pathlib import Path
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Log device usage

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_NAME = "ConvNeXtV2"
BATCH_SIZE = 8  # Used for chunk prediction
MONTHS_TO_GENERATE = [1, 3, 5, 6, 7, 8, 9, 11]

FULL_2024_PATH = "data" / "iberfire_2024.csv"
FIRERISK_MAPS_DIR = "fire_risk_map" / f"{MODEL_NAME}"
FIGURES_DIR = "images" / f"{MODEL_NAME}"
CHECKPOINTS_DIR = "checkpoints" / f"{MODEL_NAME}"
CHECKPOINT_FILE = "best_model.pt"  # Name of the saved checkpoint

# Ensure necessary directories exist
for d in [FIRERISK_MAPS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"IberFire Risk Mapping Generation Pipeline (PyTorch {MODEL_NAME})")
print("=" * 60)

# Define feature/target columns for consistency
features_to_drop = [
    "time",
    "is_fire",
    "is_near_fire",
    "x_index",
    "y_index",
    "x_coordinate",
    "y_coordinate",
    "dim_0",
    "x",
    "y",
]

# --- Feature Columns Reference (LOADED FROM A TEMPORARY FILE IF NEEDED) ---
# NOTE: The original script relies on X_train.columns for reindexing.
# To make this script fully independent for inference, you should ideally save
# the list of feature columns from the training script and load it here.
# For simplicity, we'll try to load a sample file to get the columns,
# but a definitive column list is the most robust solution.
try:
    # Load a small sample to get the correct column names and order
    # Assuming 'iberfire_train.csv' exists and has the correct feature set
    temp_data = pd.read_csv("iberfire_train.csv", nrows=1)
    X_ref = temp_data.drop(columns=features_to_drop, errors="ignore")
    feature_columns = X_ref.columns.tolist()
    print(f"1. Loaded {len(feature_columns)} feature columns from reference.")
except Exception as e:
    print(f"ERROR: Could not load reference features to define column order. {e}")
    print("Using a placeholder list, map generation may fail due to mismatch.")
    # Fallback/Placeholder (A proper saved feature list is highly recommended)
    feature_columns = []  # This needs to be correctly defined for reindexing


# ==============================================================================
# Dataset & Preprocessing (Re-used)
# ==============================================================================
def tabular_to_image(X):
    n_samples, n_features = X.shape
    side = int(np.ceil(np.sqrt(n_features)))
    padded = np.zeros((n_samples, side * side), dtype=np.float32)
    padded[:, :n_features] = X
    images = padded.reshape(n_samples, 1, side, side)
    images = np.repeat(images, 3, axis=1)
    return torch.tensor(images, dtype=torch.float32)


class FireRiskDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = tabular_to_image(X)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==============================================================================
# Model Definition (Re-used)
# ==============================================================================
class ConvNeXtV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_tiny", pretrained=True, num_classes=0
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        if x.ndim == 4:
            x = x.mean(dim=[2, 3])  # Global average pooling over H and W
        return self.fc(x)


# ==============================================================================
# Checkpoint Loading
# ==============================================================================
CHECKPOINT_PATH = CHECKPOINTS_DIR / CHECKPOINT_FILE
model = ConvNeXtV2Classifier().to(device)

print(f"2. Loading checkpoint from: {CHECKPOINT_PATH}")
try:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()  # Set model to evaluation mode
    print("   -> Checkpoint loaded successfully. Model ready for inference.")
except FileNotFoundError:
    print(
        f"ERROR: Checkpoint file not found at {CHECKPOINT_PATH}. Cannot generate maps."
    )
    exit()
except Exception as e:
    print(f"ERROR: Failed to load model state: {e}")
    exit()
print("-" * 60)


# ==============================================================================
# Utility Functions for Map Generation
# ==============================================================================
def assign_popdens(row):
    year = row["year"]
    col = f"popdens_{year}"
    return row.get(col, np.nan)


def assign_clc(row):
    year = row["year"]
    if year < 2012:
        return row["CLC_2006_forest_proportion"]
    elif year < 2018:
        return row["CLC_2012_forest_proportion"]
    else:
        return row["CLC_2018_forest_proportion"]


def process_chunk_for_prediction(chunk_df):
    chunk_df = chunk_df[chunk_df["is_spain"] == 1].copy()
    if chunk_df.empty:
        return None, None
    chunk_df["time"] = pd.to_datetime(chunk_df["time"])
    chunk_df["year"] = chunk_df["time"].dt.year
    chunk_df["month"] = chunk_df["time"].dt.month
    if "popdens" not in chunk_df.columns and any(
        c.startswith("popdens_") for c in chunk_df.columns
    ):
        chunk_df["popdens"] = chunk_df.apply(assign_popdens, axis=1)
    chunk_df["CLC_forest"] = chunk_df.apply(assign_clc, axis=1)
    metadata = chunk_df[["time", "x_coordinate", "y_coordinate", "month"]].copy()
    chunk_df = pd.get_dummies(
        chunk_df, columns=["AutonomousCommunities", "month"], dtype=int
    )
    return chunk_df.drop(
        columns=features_to_drop + ["is_spain"], errors="ignore"
    ), metadata


def create_geotiff(risk_values, coords, output_path, crs="EPSG:3035"):
    unique_x = sorted(coords["x_coordinate"].unique())
    unique_y = sorted(coords["y_coordinate"].unique(), reverse=True)
    grid = np.full((len(unique_y), len(unique_x)), np.nan, dtype=np.float32)
    x_map = {x: i for i, x in enumerate(unique_x)}
    y_map = {y: i for i, y in enumerate(unique_y)}

    # Use itertuples() and enumerate to get a 0-based position for risk_values
    for pos, row in tqdm(
        enumerate(coords.itertuples(index=False)),
        total=len(coords),
        desc="Filling Grid",
        leave=False,
    ):
        x = row.x_coordinate
        y = row.y_coordinate
        if x in x_map and y in y_map:
            grid[y_map[y], x_map[x]] = float(risk_values[pos])

    transform = from_bounds(
        min(unique_x) - 500,
        min(unique_y) - 500,
        max(unique_x) + 500,
        max(unique_y) + 500,
        len(unique_x),
        len(unique_y),
    )
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=1,
        dtype=grid.dtype,
        crs=CRS.from_string(crs),
        transform=transform,
        compress="lzw",
        nodata=np.nan,
    ) as dst:
        dst.write(grid, 1)
    return grid.shape


# ==============================================================================
# Fire Risk Map Generation
# ==============================================================================
def generate_maps():
    print("3. Generating 2024 fire risk maps from full dataset...")
    monthly_data = {m: {"metadata": []} for m in range(1, 13)}

    try:
        # Read the full dataset with chunking
        chunk_reader = pd.read_csv(FULL_2024_PATH, delimiter=";", chunksize=100000)

        for i, chunk_df in enumerate(chunk_reader):
            chunk_num = i + 1
            print(f"   -> Processing chunk {chunk_num} ({len(chunk_df)} rows)...")

            # Preprocess the chunk
            features, metadata = process_chunk_for_prediction(chunk_df)
            if features is None:
                print(
                    f"      - Chunk {chunk_num} contains no valid Spanish data. Skipping."
                )
                continue

            # Reindex to ensure feature order matches training data
            if feature_columns:
                features = features.reindex(columns=feature_columns, fill_value=0)
            else:
                # WARNING: If feature_columns is not loaded, this part may fail.
                print("WARNING: Feature column order not guaranteed to match training.")

            X_tensor = tabular_to_image(features.values).to(device)

            # Predict on the chunk
            with torch.no_grad():
                chunk_dataset = FireRiskDataset(features.values)
                chunk_loader = DataLoader(
                    chunk_dataset, batch_size=BATCH_SIZE, shuffle=False
                )

                all_logits = []
                predict_bar = tqdm(
                    chunk_loader,
                    desc=f"Chunk {chunk_num} Predict",
                    leave=False,
                    unit="batch",
                )

                for X_batch in predict_bar:
                    X_batch = X_batch.to(device)
                    # Resize input for ConvNeXt (as done in the original script)
                    X_batch = F.interpolate(
                        X_batch, size=(224, 224), mode="bilinear", align_corners=False
                    )
                    outputs = model(X_batch).squeeze()
                    all_logits.append(outputs.cpu())

                logits = torch.cat(all_logits)
                risk_proba = torch.sigmoid(logits).numpy()

            metadata["risk"] = risk_proba

            # Group and store results by month
            for month, group in metadata.groupby("month"):
                monthly_data[month]["metadata"].append(group)

        print("   -> All data chunks processed. Starting GeoTIFF creation...")

        # Use tqdm for the GeoTIFF generation loop
        for month in tqdm(MONTHS_TO_GENERATE, desc="Creating GeoTIFFs", unit="month"):
            month_name = datetime(2024, month, 1).strftime("%B")

            if not monthly_data[month]["metadata"]:
                print(f"      - No data for {month_name}. Skipping GeoTIFF.")
                continue

            # Combining data for the month
            combined_metadata = pd.concat(
                monthly_data[month]["metadata"], ignore_index=True
            )
            print(
                f"      - Aggregating {len(combined_metadata)} data points for {month_name}..."
            )

            # Aggregate by averaging risk across all data points for the same coordinate
            coord_groups = (
                combined_metadata.groupby(["x_coordinate", "y_coordinate"])
                .agg(risk=("risk", "mean"))
                .reset_index()
            )

            output_filename = FIRERISK_MAPS_DIR / f"fire_risk_2024_{month:02d}.tif"

            grid_shape = create_geotiff(
                coord_groups["risk"].values,
                coord_groups[["x_coordinate", "y_coordinate"]],
                output_filename,
            )
            print(f"      - Generated {output_filename.name} (shape: {grid_shape})")

    except Exception as e:
        print(f"\n ERROR during GeoTIFF generation: {e}")

    print("-" * 60)
    return


# ==============================================================================
# PNG Summary Map Generation
# ==============================================================================
def generate_summary_png():
    print("4. Generating PNG summary map...")
    expected_files = [
        FIRERISK_MAPS_DIR / f"fire_risk_2024_{month:02d}.tif"
        for month in MONTHS_TO_GENERATE
    ]
    tiff_paths = [path for path in expected_files if path.exists()]

    if not tiff_paths:
        print("   -> WARNING: No GeoTIFF files found to create a summary map.")
    else:
        print(f"   -> Found {len(tiff_paths)} GeoTIFF files for summary.")
        fig, axes = plt.subplots(2, 4, figsize=(16, 11), constrained_layout=True)
        axes = axes.flatten()

        # Use tqdm for the plotting loop
        for i, path in tqdm(
            enumerate(tiff_paths),
            total=len(tiff_paths),
            desc="Plotting Maps",
            unit="map",
        ):
            with rasterio.open(path) as src:
                # The crop is based on the original script's plot area
                # Adjust if needed for a different region
                raster = src.read(1)[30:900, 100:1090]

            month_name = datetime.strptime(path.stem.split("_")[-1], "%m").strftime(
                "%B"
            )
            ax = axes[i]
            # Use 'RdYlGn_r' (Red-Yellow-Green reversed) for risk visualization
            im = ax.imshow(raster, cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_title(month_name, fontsize=12)
            ax.axis("off")

        # Turn off any unused subplots
        for j in range(len(tiff_paths), len(axes)):
            axes[j].axis("off")

        # Add colorbar
        cbar = fig.colorbar(
            im, ax=axes.tolist(), orientation="horizontal", fraction=0.03, pad=0.04
        )
        cbar.set_label("Fire Risk Level (0 = Low, 1 = High)")
        output_png_path = FIGURES_DIR / "firerisks_monthly_overview.png"
        plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
        print(f"   -> Saved summary map: {output_png_path}")

    print("-" * 60)
    return


if __name__ == "__main__":
    generate_maps()
    generate_summary_png()
    print("\n Map generation pipeline completed successfully! 🎉")
