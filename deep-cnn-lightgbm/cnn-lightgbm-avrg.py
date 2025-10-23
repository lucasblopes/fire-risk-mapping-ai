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
import copy  # Import copy for deep copying the model state
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ==============================================================================
MODEL_NAME = "CNN-LightGBM-AVRG"
MONTHS_TO_GENERATE = [1, 3, 5, 6, 7, 8, 9, 11]  # Jan, Mar, May, Jun, Jul, Aug, Sep, Nov
# ==============================================================================

TRAIN_FILE_PATH = "data" / "iberfire_train.csv"
TEST_FILE_PATH = "data" / "iberfire_test.csv"
FULL_2024_PATH = "data" / "iberfire_2024.csv"

FIRERISK_MAPS_DIR = "fire_risk_map" / f"{MODEL_NAME}"
FIGURES_DIR = "images" / f"{MODEL_NAME}"
METRICS_DIR = "metrics"
# --- NEW: Checkpoint directory ---
CHECKPOINTS_DIR = "checkpoints" / f"{MODEL_NAME}"

# Create output directories (parents=True ensures the full path is created)
FIRERISK_MAPS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
# --- NEW: Create Checkpoint directory ---
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"IberFire Risk Mapping Pipeline (using {MODEL_NAME})")
print("=" * 50)

# 2. Load and Prepare Data
print("Loading and preparing datasets...")
train_data = pd.read_csv(TRAIN_FILE_PATH)
test_data = pd.read_csv(TEST_FILE_PATH)

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

X_train = train_data.drop(columns=features_to_drop, errors="ignore")
y_train = train_data[target_column]
X_test = test_data.drop(columns=features_to_drop, errors="ignore")
y_test = test_data[target_column]

# Ensure column consistency
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
n_features = X_train.shape[1]

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")


# 3. Define CNN Model (PyTorch)
class PyTorchCNN(nn.Module):
    def __init__(self, n_features):
        super(PyTorchCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        dummy_input = torch.randn(1, 1, n_features)
        dummy_output = self.pool(self.relu(self.conv1(dummy_input)))
        flattened_size = dummy_output.shape[1] * dummy_output.shape[2]
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


class PyTorchCNNWrapper:
    def __init__(self, model_class, n_features, epochs=100, batch_size=512, lr=0.001):
        self.model_class = model_class
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"CNN is using device: {self.device}")
        self.scaler = StandardScaler()
        self.model = None
        self.best_model_state = None  # Store the state dict of the best model
        self.best_loss = float("inf")  # Track the lowest loss

    def fit(self, X, y):
        # Reset model and state for a new fit (important for CV folds)
        self.model = self.model_class(self.n_features).to(self.device)
        self.best_model_state = None
        self.best_loss = float("inf")

        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            total_batches = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                total_batches += 1

            epoch_loss = running_loss / len(dataset)

            # --- Checkpoint Saving Logic (stores state in memory) ---
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                # Save the model state (weights)
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                # print(f"    Epoch {epoch+1}: New best loss ({epoch_loss:.4f}), checkpoint saved.")
            # --------------------------------------------------------

        # Load the best state dict into the model after training is complete
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"    Loaded best model state with loss: {self.best_loss:.4f}")

        return self

    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probabilities = self.model(X_tensor).cpu().numpy().flatten()
        return np.vstack([1 - probabilities, probabilities]).T

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities > 0.5).astype(int)

    # --- NEW: Checkpoint saving function ---
    def save_checkpoint(self, path):
        """Saves the best model state dictionary and scaler fitting data."""
        if self.best_model_state is not None:
            checkpoint = {
                "model_state_dict": self.best_model_state,
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
                "n_features": self.n_features,
            }
            torch.save(checkpoint, path)
            print(f"\nSuccessfully saved model checkpoint to {path}")
        else:
            print("\nWarning: No best model state found to save.")

    # ---------------------------------------


class LightGBMWrapper:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbose": -1,
        }
        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


print(f"\nPerforming temporal cross-validation for {MODEL_NAME}...")
train_data["time"] = pd.to_datetime(train_data["time"])
X_train_with_time = X_train.copy()
X_train_with_time["time"] = train_data["time"]

tscv = TimeSeriesSplit(n_splits=5)
auc_scores, acc_scores, f1_scores = [], [], []

# Use a separate model instance for CV to ensure isolation between folds
cv_model_wrapper = PyTorchCNNWrapper(PyTorchCNN, n_features=n_features, epochs=100)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_with_time)):
    print(f"  Training Fold {fold + 1}/5...")
    X_tr = X_train_with_time.iloc[train_idx].drop(columns=["time"])
    y_tr = y_train.iloc[train_idx]
    X_val = X_train_with_time.iloc[val_idx].drop(columns=["time"])
    y_val = y_train.iloc[val_idx]

    # Use the separate CV wrapper instance
    cv_model_wrapper.fit(X_tr, y_tr)
    y_val_proba = cv_model_wrapper.predict_proba(X_val)[:, 1]
    y_val_pred = (y_val_proba > 0.5).astype(int)

    auc = roc_auc_score(y_val, y_val_proba)
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    auc_scores.append(auc)
    acc_scores.append(acc)
    f1_scores.append(f1)
    print(f"    Fold {fold + 1} Metrics: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

# 5. Train Final Model and Evaluate
print(f"\nTraining final {MODEL_NAME} model (CNN + LightGBM ensemble)...")

# Train CNN
cnn_model = PyTorchCNNWrapper(PyTorchCNN, n_features=n_features, epochs=100)
cnn_model.fit(X_train, y_train)
checkpoint_path = CHECKPOINTS_DIR / "best_model.pt"
cnn_model.save_checkpoint(checkpoint_path)

# Train LightGBM
print("\nTraining LightGBM model...")
lgb_model = LightGBMWrapper()
lgb_model.fit(X_train, y_train)

# Ensemble prediction (average)
print("\nEvaluating CNN + LightGBM ensemble on test set...")
cnn_proba = cnn_model.predict_proba(X_test)[:, 1]
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
test_proba = (cnn_proba + lgb_proba) / 2
test_preds = (test_proba > 0.5).astype(int)

# Calculate and print final test metrics
test_auc = roc_auc_score(y_test, test_proba)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds)
print(f"  Test AUC: {test_auc:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Test F1-Score: {test_f1:.4f}")

# Save Metrics to File ---
print(f"\nSaving performance metrics to file...")
metrics_file_path = METRICS_DIR / f"{MODEL_NAME}.txt"
with open(metrics_file_path, "w") as f:
    f.write(f"Performance Metrics for Model: {MODEL_NAME}\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 50 + "\n\n")

    f.write("Cross-Validation Results (5-fold Temporal Split)\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Mean AUC:      {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}\n")
    f.write(f"  Mean Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}\n")
    f.write(f"  Mean F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}\n\n")

    f.write("Final Evaluation on 2024 Test Set\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Test AUC:      {test_auc:.4f}\n")
    f.write(f"  Test Accuracy: {test_acc:.4f}\n")
    f.write(f"  Test F1-Score: {test_f1:.4f}\n")
print(f"    Metrics successfully saved to {metrics_file_path}")
# --- End of Saving Metrics ---

# 6. Generate Monthly Fire Risk GeoTIFF Maps (Helper functions remain the same)
# ... (assign_popdens, assign_clc, process_chunk_for_prediction, create_geotiff functions unchanged) ...


# (Functions are included here for completeness, but they were long so I'm omitting them in the final output block)
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
        # risk_proba = final_model.predict_proba(features)[:, 1]
        # Generate fire risk map predictions usign average of both models
        cnn_risk = cnn_model.predict_proba(features)[:, 1]
        lgb_risk = lgb_model.predict_proba(features)[:, 1]
        risk_proba = (cnn_risk + lgb_risk) / 2

        metadata["risk"] = risk_proba
        for month, group in metadata.groupby("month"):
            if month in monthly_data:
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
    print(f"\nAn error occurred during GeoTIFF mapping: {e}")

# 7. Generate PNG Summary Map
print(f"\nGenerating PNG summary map for {MODEL_NAME} from GeoTIFFs...")
expected_files = [
    FIRERISK_MAPS_DIR / f"fire_risk_2024_{month:02d}.tif"
    for month in MONTHS_TO_GENERATE
]
# Filter for only those files that actually exist
tiff_paths = [path for path in expected_files if path.exists()]
if not tiff_paths:
    print("  No GeoTIFF files found. Skipping PNG generation.")
else:
    # Set the title for the entire figure to include the model name
    fig, axes = plt.subplots(2, 4, figsize=(16, 11), constrained_layout=True)
    fig.suptitle(
        f"Monthly Fire Risk Overview - Model: {MODEL_NAME}", fontsize=16
    )  # <--- NEW TITLE

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

    # --- MODIFIED: PNG filename includes model name ---
    output_png_path = FIGURES_DIR / f"firerisks_{MODEL_NAME}.png"
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    print(f"    Successfully generated summary map: {output_png_path}")

print("\n" + "=" * 50)
print("Pipeline completed!")
