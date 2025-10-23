import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import os
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

# --- Model Imports ---
# Import all required model libraries here
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CHOOSE YOUR MODEL HERE
# ==============================================================================
# Options: "RandomForest", "XGBoost", "LightGBM"
MODEL_NAME = "XGBoost"
# Define the specific months for the second summary map
MONTHS_8_MAP = [1, 3, 5, 6, 7, 8, 9, 11]
# ==============================================================================

# --- Define Paths and Directories ---
TRAIN_FILE_PATH = "iberfire_train.csv"
TEST_FILE_PATH = "iberfire_test.csv"
FULL_2024_PATH = "iberfire_2024.csv"

# Output directories will be named based on the selected model
FIRERISK_MAPS_DIR = Path(f"./fire_risk_maps/{MODEL_NAME}")
FIGURES_DIR = Path(f"./images/{MODEL_NAME}")
METRICS_DIR = Path("./metrics")

# Create output directories
FIRERISK_MAPS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)


# --- Helper function for unique file paths ---
def get_unique_path(path_obj):
    """Checks if a file exists. If so, appends a number to make it unique."""
    if not path_obj.exists():
        return path_obj
    else:
        stem = path_obj.stem
        suffix = path_obj.suffix
        parent = path_obj.parent
        counter = 1
        while True:
            new_name = f"{stem}-{counter}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1


print(f"IberFire Risk Mapping Pipeline (using {MODEL_NAME})")
print("=" * 50)

# 2. Load the training and testing datasets
print("Loading training and test datasets...")
train_data = pd.read_csv(TRAIN_FILE_PATH)
test_data = pd.read_csv(TEST_FILE_PATH)

# 3. Define features to drop
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
target_column = "is_fire"

# Prepare training data
X_train = train_data.drop(columns=features_to_drop, errors="ignore")
y_train = train_data[target_column]
X_test = test_data.drop(columns=features_to_drop, errors="ignore")
y_test = test_data[target_column]

# Ensure column consistency
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print(f"Training dataset shape: {X_train.shape}")
print(f"Testing dataset shape: {X_test.shape}")

# 4. Define Models and Hyperparameters
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=123, n_jobs=-1, min_samples_leaf=10
    ),
    "XGBoost": xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=5,
        eval_metric="logloss",
        random_state=123,
        n_jobs=-1,
        verbosity=0,
        use_label_encoder=False,
    ),
    "LightGBM": lgb.LGBMClassifier(
        objective="binary", n_estimators=200, max_depth=5, random_state=123, n_jobs=-1
    ),
}

# Select the model to run
model_to_run = models.get(MODEL_NAME)
if model_to_run is None:
    raise ValueError(
        f"Model '{MODEL_NAME}' not recognized. Available options are: {list(models.keys())}"
    )

# 5. Temporal Cross-Validation
print(f"\nPerforming temporal cross-validation for {MODEL_NAME}...")
train_data["time"] = pd.to_datetime(train_data["time"])
X_train_with_time = X_train.copy()
X_train_with_time["time"] = train_data["time"]

# --- CHANGE HERE: Set n_splits to 10 ---
tscv = TimeSeriesSplit(n_splits=10)
auc_scores, acc_scores, f1_scores = [], [], []
n_splits = tscv.get_n_splits()

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_with_time, y_train)):
    X_tr = X_train_with_time.iloc[train_idx].drop(columns=["time"])
    X_val = X_train_with_time.iloc[val_idx].drop(columns=["time"])
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model_to_run.fit(X_tr, y_tr)
    y_val_proba = model_to_run.predict_proba(X_val)[:, 1]
    y_val_pred = model_to_run.predict(X_val)
    auc = roc_auc_score(y_val, y_val_proba)
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    auc_scores.append(auc)
    acc_scores.append(acc)
    f1_scores.append(f1)
    # --- CHANGE HERE: Updated print statement to be dynamic ---
    print(f"  Fold {fold + 1}/{n_splits}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

# 6. Train final model
print(f"\nTraining final {MODEL_NAME} model...")
final_model = model_to_run
final_model.fit(X_train, y_train)

# 7. Evaluate on test set and Save Metrics
print(f"\nEvaluating {MODEL_NAME} on test set...")
test_preds = final_model.predict(X_test)
test_proba = final_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_proba)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Test F1: {test_f1:.4f}")

# Save metrics to a unique file
metrics_file_path = get_unique_path(METRICS_DIR / f"{MODEL_NAME}.txt")
print(f"\nSaving performance metrics to {metrics_file_path}...")
with open(metrics_file_path, "w") as f:
    f.write(f"Performance Metrics for Model: {MODEL_NAME}\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 50 + "\n\n")
    # --- CHANGE HERE: Updated metrics file to be dynamic ---
    f.write(f"Cross-Validation Results ({n_splits}-fold Temporal Split)\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Mean AUC:      {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n")
    f.write(f"  Mean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}\n")
    f.write(f"  Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n\n")
    f.write("Final Evaluation on Test Set\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Test AUC:      {test_auc:.4f}\n")
    f.write(f"  Test Accuracy: {test_acc:.4f}\n")
    f.write(f"  Test F1-Score: {test_f1:.4f}\n")
print("   Metrics saved successfully.")


# 8. Generate monthly fire risk maps (GeoTIFF)
# (Helper functions are unchanged)
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
    if len(risk_values) == 0:
        return None
    unique_x = sorted(coords["x_coordinate"].unique())
    unique_y = sorted(coords["y_coordinate"].unique(), reverse=True)
    grid = np.full((len(unique_y), len(unique_x)), np.nan, dtype=np.float32)
    x_map = {x: i for i, x in enumerate(unique_x)}
    y_map = {y: i for i, y in enumerate(unique_y)}
    for idx, (_, row) in enumerate(coords.iterrows()):
        x, y = row["x_coordinate"], row["y_coordinate"]
        if x in x_map and y in y_map:
            grid[y_map[y], x_map[x]] = risk_values[idx]
    px_size = 1000
    transform = from_bounds(
        min(unique_x) - px_size / 2,
        min(unique_y) - px_size / 2,
        max(unique_x) + px_size / 2,
        max(unique_y) + px_size / 2,
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


print(f"\nGenerating monthly fire risk maps for 2024 using {MODEL_NAME}...")
try:
    print("  Loading 2024 data in chunks...")
    monthly_data = {m: {"metadata": [], "risks": []} for m in range(1, 13)}
    chunk_size = 100000
    for i, chunk_df in enumerate(
        pd.read_csv(FULL_2024_PATH, delimiter=";", chunksize=chunk_size)
    ):
        print(f"    Processing chunk {i + 1} ({len(chunk_df)} rows)...")
        features, metadata = process_chunk_for_prediction(chunk_df)
        if features is None:
            continue
        features = features.reindex(columns=X_train.columns, fill_value=0)
        risk_proba = final_model.predict_proba(features)[:, 1]
        metadata["risk"] = risk_proba
        for month, group in metadata.groupby("month"):
            monthly_data[month]["metadata"].append(group)
            monthly_data[month]["risks"].extend(group["risk"].values)
    print("  Creating monthly GeoTIFF files...")
    for month in range(1, 13):
        month_name = datetime(2024, month, 1).strftime("%B")
        print(f"    Generating {month_name} 2024...")
        if not monthly_data[month]["metadata"]:
            print(f"      No data for {month_name}")
            continue
        combined_metadata = pd.concat(
            monthly_data[month]["metadata"], ignore_index=True
        )
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
        if grid_shape:
            print(f"        Generated {output_filename} (shape: {grid_shape})")
except FileNotFoundError:
    print(f"\nCould not find {FULL_2024_PATH}. Skipping map generation.")
except Exception as e:
    print(f"\nError during GeoTIFF mapping: {e}")


# 9. Generate 12-Month PNG Summary Map
print(f"\nGenerating 12-month PNG summary map for {MODEL_NAME}...")
tiff_paths = sorted(FIRERISK_MAPS_DIR.glob("fire_risk_*.tif"))
if not tiff_paths:
    print("  No GeoTIFF files found. Skipping PNG generation.")
else:
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    axes = axes.flatten()
    im = None
    for i, path in enumerate(tiff_paths):
        with rasterio.open(path) as src:
            raster = src.read(1)[30:900, 100:1090]
        month_name = datetime.strptime(path.stem.split("_")[-1], "%m").strftime("%B")
        ax = axes[i]
        im = ax.imshow(raster, cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_title(month_name, fontsize=12)
        ax.axis("off")
    for j in range(len(tiff_paths), len(axes)):
        axes[j].axis("off")
    if im:
        cbar = fig.colorbar(
            im, ax=axes.tolist(), orientation="horizontal", fraction=0.03, pad=0.04
        )
        cbar.set_label("Fire Risk Level (0 = Low, 1 = High)")
    output_png_path = get_unique_path(
        FIGURES_DIR / "firerisks_monthly_overview_12_months.png"
    )
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    print(f"   Successfully generated 12-month map: {output_png_path}")
    plt.close(fig)  # Close the figure to free up memory

# 10. Generate 8-Month PNG Summary Map
print(f"\nGenerating 8-month PNG summary map for {MODEL_NAME}...")
tiff_paths_8_months = [
    p for p in tiff_paths if int(p.stem.split("_")[-1]) in MONTHS_8_MAP
]
if not tiff_paths_8_months:
    print("  No GeoTIFF files for the specified 8 months found. Skipping.")
else:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes = axes.flatten()
    im = None
    for i, path in enumerate(tiff_paths_8_months):
        with rasterio.open(path) as src:
            raster = src.read(1)[30:900, 100:1090]
        month_name = datetime.strptime(path.stem.split("_")[-1], "%m").strftime("%B")
        ax = axes[i]
        im = ax.imshow(raster, cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_title(month_name, fontsize=12)
        ax.axis("off")
    for j in range(len(tiff_paths_8_months), len(axes)):
        axes[j].axis("off")
    if im:
        cbar = fig.colorbar(
            im, ax=axes.tolist(), orientation="horizontal", fraction=0.04, pad=0.05
        )
        cbar.set_label("Fire Risk Level (0 = Low, 1 = High)")
    output_png_path = get_unique_path(
        FIGURES_DIR / "firerisks_monthly_overview_8_months.png"
    )
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    print(f"   Successfully generated 8-month map: {output_png_path}")
    plt.close(fig)

print("\n" + "=" * 50)
print("Pipeline completed!")
