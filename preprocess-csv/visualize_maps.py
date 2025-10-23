import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Define directories as Path objects
FIRERISK_MAPS_DIR = Path("./fire_risk_maps_2024")
FIGURES_DIR = Path("./images")

# Ensure the output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# List of new file paths
paths = [
    FIRERISK_MAPS_DIR / "fire_risk_2024_01.tif",
    FIRERISK_MAPS_DIR / "fire_risk_2024_03.tif",
    FIRERISK_MAPS_DIR / "fire_risk_2024_05.tif",
    FIRERISK_MAPS_DIR / "fire_risk_2024_06.tif",
    FIRERISK_MAPS_DIR / "fire_risk_2024_07.tif",
    FIRERISK_MAPS_DIR / "fire_risk_2024_08.tif",
    FIRERISK_MAPS_DIR / "fire_risk_2024_09.tif",
    FIRERISK_MAPS_DIR / "fire_risk_2024_11.tif",
]

fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
axes = axes.flatten()

for i, path in enumerate(paths):
    with rasterio.open(path) as src:
        raster = src.read(1)
        raster = raster[30:900, 100:1090]

    filename = path.stem

    # FIX: Parse the new filename format (e.g., "fire_risk_2024_01")
    parts = filename.split("_")
    year = parts[2]
    month = parts[3]
    date_str = f"{year}-{month}"  # Create a string like "2024-01"

    # Use the correct format to get the month name
    month_name = datetime.strptime(date_str, "%Y-%m").strftime("%B")

    ax = axes[i]
    im = ax.imshow(raster, cmap="RdYlGn_r", vmin=0, vmax=1)

    # FIX: Update the title to be just the month name
    ax.set_title(month_name, fontsize=10)
    ax.axis("off")

# Adjust layout to prevent title overlap
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.02, pad=0.04)
cbar.set_label("Fire risk level (0 = low, 1 = high)")

plt.savefig(FIGURES_DIR / "firerisks.png", dpi=300, bbox_inches="tight", format="png")
