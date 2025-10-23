import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
FILE_PATH = "./IberFire.nc"


def plot_static_map(ds: xr.Dataset, var_name: str):
    """
    Visualizes 2D spatial-only data.
    Example: elevation_mean, is_spain
    """
    try:
        data = ds[var_name].where(ds["is_spain"] == 1)

        plt.figure(figsize=(10, 8))
        data.plot(cmap="viridis", cbar_kwargs={"label": data.attrs.get("units", "")})
        plt.title(f"Static Map of: {var_name}")
        plt.xlabel("X Coordinate (EPSG:3035)")
        plt.ylabel("Y Coordinate (EPSG:3035)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    except Exception as e:
        print(f"Error plotting static map: {e}")


def plot_map_at_time(ds: xr.Dataset, var_name: str):
    """
    Visualizes a 2D map of spatio-temporal data at a user-specified time.
    Example: t2m_mean on '2024-08-01'
    """
    date = input("Enter a date (YYYY-MM-DD) to visualize: ")
    try:
        data_slice = ds[var_name].sel(time=date, method="nearest")
        data_slice = data_slice.where(ds["is_spain"] == 1)

        plt.figure(figsize=(10, 8))
        data_slice.plot(
            cmap="inferno", cbar_kwargs={"label": data_slice.attrs.get("units", "")}
        )
        plt.title(f"{var_name} on {date}")
        plt.xlabel("X Coordinate (EPSG:3035)")
        plt.ylabel("Y Coordinate (EPSG:3035)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    except KeyError:
        print(
            f"Date '{date}' not found. Please use a valid date within the dataset's range."
        )
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_timeseries_at_point(ds: xr.Dataset, var_name: str):
    """
    Plots a time-series graph for a variable at a user-specified spatial point.
    """
    try:
        max_x = ds.dims["x"] - 1
        max_y = ds.dims["y"] - 1
        x_idx = int(input(f"Enter X index [0-{max_x}]: "))
        y_idx = int(input(f"Enter Y index [0-{max_y}]: "))

        if not (0 <= x_idx <= max_x and 0 <= y_idx <= max_y):
            print("Error: Index out of bounds.")
            return

        # Check if the selected point is in Spain for meaningful data
        if ds["is_spain"].isel(x=x_idx, y=y_idx).item() == 0:
            print("Warning: Selected point is outside of Spain.")

        timeseries = ds[var_name].isel(x=x_idx, y=y_idx)

        plt.figure(figsize=(12, 6))
        timeseries.plot()
        plt.title(f"Time-Series of {var_name} at Point (x={x_idx}, y={y_idx})")
        plt.xlabel("Date")
        plt.ylabel(f"{var_name} ({timeseries.attrs.get('units', '')})")
        plt.grid(True)
        plt.show()

    except ValueError:
        print("Invalid input. Please enter integer indices.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """
    Main function to run the interactive visualization menu.
    """
    try:
        with xr.open_dataset(FILE_PATH) as ds:
            print(f"Successfully loaded '{FILE_PATH}'")

            while True:
                print("\n--- IberFire Data Visualizer ---")
                uncompressed_size_bytes = ds.nbytes
                uncompressed_size_gb = uncompressed_size_bytes / (1024**3)
                print(
                    f" Uncompressed dataset size in memory: {uncompressed_size_gb:.2f} GB"
                )

                variables = list(ds.data_vars)
                for i, var in enumerate(variables):
                    print(f"{i:3d}: {var}")

                try:
                    choice = input(
                        "\nSelect a variable number to visualize (or 'q' to quit): "
                    )
                    if choice.lower() == "q":
                        break
                    var_idx = int(choice)
                    var_name = variables[var_idx]
                    selected_var = ds[var_name]
                    print(f"\nSelected '{var_name}'...")

                    # --- Determine Data Type and Offer Options ---
                    is_spatiotemporal = "time" in selected_var.dims
                    is_spatial = "x" in selected_var.dims and "y" in selected_var.dims

                    if is_spatiotemporal and is_spatial:
                        print("This is Spatio-Temporal data. Choose a visualization:")
                        print("  1: Plot a map for a specific date")
                        print("  2: Plot a time-series for a specific point")
                        plot_choice = input("Enter your choice (1 or 2): ")
                        if plot_choice == "1":
                            plot_map_at_time(ds, var_name)
                        elif plot_choice == "2":
                            plot_timeseries_at_point(ds, var_name)
                        else:
                            print("Invalid choice.")

                    elif is_spatial:
                        print("This is Spatial-only data. Displaying static map.")
                        plot_static_map(ds, var_name)

                    else:
                        print("This data format is not supported by the visualizer.")

                except (ValueError, IndexError):
                    print("Invalid input. Please enter a valid number from the list.")

    except FileNotFoundError:
        print(f" ERROR: The file '{FILE_PATH}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
