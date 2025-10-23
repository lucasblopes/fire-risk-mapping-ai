import xarray as xr
import pandas as pd
import numpy as np

# ================================
# File paths
# ================================
NC_PATH = "IberFire.nc"  # original NetCDF datacube
OUT_CSV = "iberfire_2024.csv"

# ================================
# 1. Load IberFire datacube
# ================================
ds = xr.open_dataset(NC_PATH)

# Ensure time is parsed as datetime
if not np.issubdtype(ds["time"].dtype, np.datetime64):
    ds["time"] = pd.to_datetime(ds["time"].values)

# ================================
# 2. Filter 2024 & Spain only
# ================================
mask_2024 = ds["time.year"] == 2024
mask_spain = ds["is_spain"] == 1

subset = ds.where(mask_2024 & mask_spain, drop=True)

# ================================
# 3. Handle population density (popdens)
# ================================
# Create a new variable 'popdens' based on correct year
popdens_vars = [v for v in subset.data_vars if v.startswith("popdens_")]
popdens_years = [int(v.split("_")[1]) for v in popdens_vars]

popdens_values = []
for t in subset["time"].values:
    year = pd.to_datetime(str(t)).year
    # Find closest available popdens year (<= current year)
    available = [y for y in popdens_years if y <= year]
    if not available:
        chosen = min(popdens_years)
    else:
        chosen = max(available)
    varname = f"popdens_{chosen}"
    popdens_values.append(subset[varname].expand_dims(time=[t]))

# Concatenate into single DataArray
popdens_da = xr.concat(popdens_values, dim="time")
subset = subset.assign(popdens=popdens_da)

# ================================
# 4. Handle CLC year-dependent variables
# ================================
# Choose appropriate CLC year (2006, 2012, 2018)

CLC_years = [2006, 2012, 2018]


def get_clc_year(date):
    if date.year < 2012:
        return 2006
    elif date.year < 2018:
        return 2012
    else:
        return 2018


clc_vars = [v for v in subset.data_vars if v.startswith("CLC_")]

# Example: take only the correct year’s proportion features
# (here we simplify: user may refine later)

selected_clc = []
for t in subset["time"].values:
    year = get_clc_year(pd.to_datetime(str(t)))
    vars_for_year = [v for v in clc_vars if f"CLC_{year}" in v]
    selected_clc.append(subset[vars_for_year].expand_dims(time=[t]))

clc_da = xr.concat(selected_clc, dim="time")
subset = subset.assign(**{var: clc_da[var] for var in clc_da.data_vars})

# ================================
# 5. Convert to DataFrame
# ================================
df = subset.to_dataframe().reset_index()

# Drop any rows with NaNs (e.g. sea cells)
df = df.dropna()

# Save as CSV
print(f"Saving {OUT_CSV} with shape {df.shape}")
df.to_csv(OUT_CSV, sep=";", index=False)
