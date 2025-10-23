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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tqdm import tqdm  # Import tqdm for progress bars

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Log device usage

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_NAME = "ConvNeXtV2"
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-4
MONTHS_TO_GENERATE = [1, 3, 5, 6, 7, 8, 9, 11]

TRAIN_FILE_PATH = "iberfire_train.csv"
TEST_FILE_PATH = "iberfire_test.csv"
FULL_2024_PATH = "iberfire_2024.csv"

FIRERISK_MAPS_DIR = Path(f"./fire_risk_maps/{MODEL_NAME}")
FIGURES_DIR = Path(f"./images/{MODEL_NAME}")
METRICS_DIR = Path("./metrics")
CHECKPOINT_DIR = Path(f"./checkpoints/{MODEL_NAME}")

for d in [FIRERISK_MAPS_DIR, FIGURES_DIR, METRICS_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"IberFire Risk Mapping Pipeline (PyTorch {MODEL_NAME})")
print(f"Configuration: Epochs={EPOCHS}, Batch Size={BATCH_SIZE}, LR={LR}")
print("=" * 60)

# ==============================================================================
# Data Loading
# ==============================================================================
print("1. Loading datasets...")
try:
    train_data = pd.read_csv(TRAIN_FILE_PATH)
    test_data = pd.read_csv(TEST_FILE_PATH)
    print(f"   -> Train data loaded: {len(train_data)} samples.")
    print(f"   -> Test data loaded: {len(test_data)} samples.")
except Exception as e:
    print(f"ERROR: Could not load datasets. Check file paths. {e}")
    exit()

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
y_train = train_data[target_column].values.astype(np.float32)
X_test = test_data.drop(columns=features_to_drop, errors="ignore")
y_test = test_data[target_column].values.astype(np.float32)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
feature_count = X_train.shape[1]
print(f"   -> Feature count after preprocessing: {feature_count}")
print("-" * 60)


# ==============================================================================
# Dataset & Preprocessing
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
        # Add a log for image conversion
        print("   -> Converting tabular data to 'image' tensors...")
        self.X = tabular_to_image(X)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        print(f"   -> Data shape: {self.X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ==============================================================================
# Model Definition
# ==============================================================================
class ConvNeXtV2Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Log model initialization
        print("2. Initializing ConvNeXtV2Classifier...")
        self.backbone = timm.create_model(
            "convnextv2_tiny", pretrained=True, num_classes=0
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.backbone.num_features, 1)
        print(f"   -> Backbone loaded with {self.backbone.num_features} features.")

    def forward(self, x):
        x = self.backbone(x)
        # Flatten global features
        if x.ndim == 4:
            x = x.mean(dim=[2, 3])  # Global average pooling over H and W
        return self.fc(x)


# ==============================================================================
# Training & Evaluation
# ==============================================================================
# ==============================================================================
# Training & Evaluation (best-model only)
# ==============================================================================
def train_model(model, train_loader, criterion, optimizer, epoch_start=0):
    model.train()
    best_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"

    loss_history = []

    for epoch in range(epoch_start, EPOCHS):
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")

        for X_batch, y_batch in train_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # Resize input for ConvNeXt
            X_batch = F.interpolate(
                X_batch, size=(224, 224), mode="bilinear", align_corners=False
            )
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Update tqdm with live loss info
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"\n🧠 Epoch [{epoch + 1}/{EPOCHS}] | Average Loss: {avg_loss:.4f}")

        # --- Save best model if improved ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_ckpt_path)
            print(
                f"💾 New best model saved! Epoch {best_epoch} | Loss: {best_loss:.4f}"
            )

    # --- Plot and save loss curve ---
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, label="Training Loss", color="tab:red")
    plt.axvline(
        best_epoch, color="tab:blue", linestyle="--", label=f"Best Epoch {best_epoch}"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve ({MODEL_NAME})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{MODEL_NAME}_loss_curve.png")
    print(
        f"\n📉 Training loss curve saved to: {FIGURES_DIR / f'{MODEL_NAME}_loss_curve.png'}"
    )
    print(f"🏆 Best model from Epoch {best_epoch} with Loss = {best_loss:.4f}")


# Data prep for training
train_dataset = FireRiskDataset(X_train.values, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = ConvNeXtV2Classifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

print("-" * 60)
print("3. Starting Model Training...")
train_model(model, train_loader, criterion, optimizer)
print("Training complete.")
print("-" * 60)

# ==============================================================================
# Evaluation
# ==============================================================================
print("4. Evaluating model performance on test set...")
model.eval()
with torch.no_grad():
    # Use tqdm for the prediction loop
    test_tensor = tabular_to_image(X_test.values)
    test_dataset = FireRiskDataset(X_test.values, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_logits = []

    # Predict in batches to prevent memory issues and use a progress bar
    test_bar = tqdm(test_loader, desc="Testing", unit="batch")
    for X_batch, _ in test_bar:
        X_batch = X_batch.to(device)
        X_batch = F.interpolate(
            X_batch, size=(224, 224), mode="bilinear", align_corners=False
        )
        outputs = model(X_batch).squeeze()
        all_logits.append(outputs.cpu())

    logits = torch.cat(all_logits)
    probs = torch.sigmoid(logits).numpy()

y_pred = (probs > 0.5).astype(int)
auc = roc_auc_score(y_test, probs)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("   -> Evaluation Results:")
print(f"      - Test Accuracy: {acc:.4f}")
print(f"      - Test AUC: {auc:.4f}")
print(f"      - Test F1: {f1:.4f}")

with open(METRICS_DIR / f"{MODEL_NAME}.txt", "w") as f:
    f.write(f"Model: {MODEL_NAME}\nGenerated: {datetime.now()}\n")
    f.write(f"Accuracy: {acc:.4f}\nAUC: {auc:.4f}\nF1: {f1:.4f}\n")
print(f"   -> Metrics saved to {METRICS_DIR / f'{MODEL_NAME}.txt'}")
print("-" * 60)


# ==============================================================================
# Fire Risk Map Generation
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


try:
    print("5. Generating 2024 fire risk maps from full dataset...")
    monthly_data = {m: {"metadata": [], "risks": []} for m in range(1, 13)}

    # Read the full dataset with chunking and use tqdm for chunk iteration
    chunk_reader = pd.read_csv(FULL_2024_PATH, delimiter=";", chunksize=100000)

    # Estimate total number of chunks (requires first iteration to get reader size,
    # but for simplicity, we'll just track the chunk number)

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

        print(f"      - Preprocessed features shape: {features.shape}")

        features = features.reindex(columns=X_train.columns, fill_value=0)
        X_tensor = tabular_to_image(features.values).to(device)

        # Predict on the chunk
        with torch.no_grad():
            # Use tqdm for the prediction batch loop within the chunk
            chunk_dataset = FireRiskDataset(
                features.values
            )  # Recreate dataset for chunk
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

        coord_groups = (
            combined_metadata.groupby(["x_coordinate", "y_coordinate"])
            .agg(risk=("risk", "mean"))
            .reset_index()
        )

        output_filename = FIRERISK_MAPS_DIR / f"fire_risk_2024_{month:02d}.tif"

        # The 'create_geotiff' function now has an internal progress bar for grid filling
        grid_shape = create_geotiff(
            coord_groups["risk"].values,
            coord_groups[["x_coordinate", "y_coordinate"]],
            output_filename,
        )
        print(f"      - Generated {output_filename.name} (shape: {grid_shape})")

except Exception as e:
    print(f"\n Error during GeoTIFF generation: {e}")

print("-" * 60)

# ==============================================================================
# PNG Summary Map
# ==============================================================================
print("6. Generating PNG summary map...")
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
        enumerate(tiff_paths), total=len(tiff_paths), desc="Plotting Maps", unit="map"
    ):
        with rasterio.open(path) as src:
            raster = src.read(1)[30:900, 100:1090]

        month_name = datetime.strptime(path.stem.split("_")[-1], "%m").strftime("%B")
        ax = axes[i]
        im = ax.imshow(raster, cmap="RdYlGn_r", vmin=0, vmax=1)
        ax.set_title(month_name, fontsize=12)
        ax.axis("off")

    for j in range(len(tiff_paths), len(axes)):
        axes[j].axis("off")

    cbar = fig.colorbar(
        im, ax=axes.tolist(), orientation="horizontal", fraction=0.03, pad=0.04
    )
    cbar.set_label("Fire Risk Level (0 = Low, 1 = High)")
    output_png_path = FIGURES_DIR / "firerisks_monthly_overview.png"
    plt.savefig(output_png_path, dpi=300, bbox_inches="tight")
    print(f"   -> Saved summary map: {output_png_path}")

print("\n Pipeline completed successfully!")
