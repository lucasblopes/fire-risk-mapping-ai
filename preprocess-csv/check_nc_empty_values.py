import xarray as xr


def check_spain_variables_memory_safe(file_path):
    """
    Checks for NaN values in all variables within the Spanish territory of a NetCDF file.
    This version computes the boolean mask first to avoid dask errors.

    Args:
        file_path (str): The path to the .nc file.
    """
    try:
        # Open the dataset with chunks to handle large files
        ds = xr.open_dataset(file_path, chunks="auto")

        print(f"--- Checking file: {file_path} (Memory-Safe Mode | Spain Only) ---")

        if "is_spain" not in ds:
            print(
                "Error: 'is_spain' variable not found. Cannot filter for Spanish territory."
            )
            return

        # **CORRECTION**: Load the small 'is_spain' mask into memory first.
        # This is a small, 2D spatial variable, so it's safe to load.
        is_spain_mask = ds.is_spain.load()

        # Now, filter the full (lazy) dataset using the computed mask.
        spain_ds = ds.where(is_spain_mask == 1, drop=True)

        print("Successfully filtered for Spanish territory. Now checking variables...")

        # Loop through all data variables in the filtered dataset
        for var_name in spain_ds.data_vars:
            if var_name == "is_spain":
                continue

            variable = spain_ds[var_name]

            print(f"Processing variable: '{var_name}'...")

            # Count the number of NaN values in the chunked data
            num_nan = variable.isnull().sum().compute()

            count = num_nan.item()

            if count > 0:
                print(f"Found {count} empty value(s) in variable: '{var_name}'")
            else:
                print(f"No empty values found in variable: '{var_name}'")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Example Usage ---
# Replace 'your_file.nc' with the path to your IberFire NetCDF file.
check_spain_variables_memory_safe("../IberFire/IberFire.nc")
