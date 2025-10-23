# Import the necessary library
import xarray as xr

# --- Configuration ---
# Define the input and output file paths
input_filepath = "IberFire.nc"
output_filepath = "IberFire_May_to_Sep.nc"

print(f" Loading dataset: {input_filepath}...")

# Use a try-except block for robust error handling
try:
    # Open the NetCDF dataset
    # The 'time' coordinate is automatically parsed into datetime objects
    ds = xr.open_dataset(input_filepath)
    print(" Dataset loaded successfully.")
    print("\nOriginal dataset dimensions:")
    print(ds.dims)

    # --- Slicing Logic ---
    # Create a boolean mask to identify the time entries where the month is
    # between May (5) and September (9), inclusive.
    # We access the month component via the '.dt' accessor on the time coordinate.
    month_mask = (ds["time"].dt.month >= 5) & (ds["time"].dt.month <= 9)
    print(month_mask)

    # Apply the mask to the 'time' dimension to select only the desired months
    print("\n🔪 Slicing data for months May through September...")
    ds_sliced = ds.sel(time=month_mask)
    print(" Slicing complete.")

    # --- Save the Result ---
    print(f"\n Saving sliced dataset to: {output_filepath}...")
    # Save the new dataset to a NetCDF file
    # The 'encoding' parameter helps ensure compression is handled well
    ds_sliced.to_netcdf(output_filepath)
    print(" New file saved successfully!")

    print("\nSliced dataset dimensions:")
    print(ds_sliced.dims)


except FileNotFoundError:
    print(
        f" ERROR: The file '{input_filepath}' was not found. Please ensure it's in the correct directory."
    )
except KeyError:
    print(
        " ERROR: Could not find the 'time' coordinate. Please check the dataset for the correct time dimension name (e.g., 't', 'date')."
    )
except Exception as e:
    print(f" An unexpected error occurred: {e}")

finally:
    # Close the dataset to free up resources, if it was opened
    if "ds" in locals():
        ds.close()
    print("\nScript finished.")
