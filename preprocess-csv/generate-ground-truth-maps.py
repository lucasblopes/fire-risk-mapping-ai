"""
Generate ground truth monthly fire maps from 2024 data (no model).
Uses is_fire variable directly from IberFire.nc.
"""

import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

NC_PATH = "IberFire.nc"
OUTPUT_DIR = Path("ground_truth_maps_2024")
MONTHS = [1, 3, 5, 6, 7, 8, 9, 11]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ds = xr.open_dataset(NC_PATH)
ds_2024 = ds.sel(time=ds["time.year"] == 2024)
spain_mask = ds["is_spain"].values == 1

print(f"Loaded dataset. 2024 timesteps: {len(ds_2024.time)}")

for month in MONTHS:
    ds_month = ds_2024.sel(time=ds_2024["time.month"] == month)
    if len(ds_month.time) == 0:
        print(f"  No data for month {month}, skipping.")
        continue

    # Ground truth: fire occurred at least once in the month
    fire_grid = ds_month["is_fire"].max(dim="time").values.astype(np.float32)
    fire_grid = np.where(spain_mask, fire_grid, np.nan)

    # Write GeoTIFF
    x_coords = ds["x_coordinate"].values
    y_coords = ds["y_coordinate"].values
    transform = from_bounds(
        np.nanmin(x_coords) - 500, np.nanmin(y_coords) - 500,
        np.nanmax(x_coords) + 500, np.nanmax(y_coords) + 500,
        fire_grid.shape[1], fire_grid.shape[0],
    )
    out_path = OUTPUT_DIR / f"ground_truth_2024_{month:02d}.tif"
    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=fire_grid.shape[0], width=fire_grid.shape[1],
        count=1, dtype="float32",
        crs=CRS.from_string("EPSG:3035"),
        transform=transform, compress="lzw", nodata=np.nan,
    ) as dst:
        dst.write(fire_grid, 1)

    n_fire = int(np.nansum(fire_grid))
    month_name = datetime(2024, month, 1).strftime("%B")
    print(f"  {month_name}: {n_fire} fire cells -> {out_path.name}")

# Summary PNG
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
for i, month in enumerate(MONTHS):
    tif = OUTPUT_DIR / f"ground_truth_2024_{month:02d}.tif"
    with rasterio.open(tif) as src:
        raster = src.read(1)
    ax = axes.flatten()[i]
    im = ax.imshow(raster, cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_title(datetime(2024, month, 1).strftime("%B"))
    ax.axis("off")

for j in range(len(MONTHS), 8):
    axes.flatten()[j].axis("off")

fig.colorbar(im, ax=axes.tolist(), orientation="horizontal", fraction=0.03, pad=0.04,
             label="Fire Occurrence (0=No Fire, 1=Fire)")
plt.savefig(OUTPUT_DIR / "ground_truth_2024_summary.png", dpi=300, bbox_inches="tight")
print(f"\nDone. Maps saved to {OUTPUT_DIR}/")
