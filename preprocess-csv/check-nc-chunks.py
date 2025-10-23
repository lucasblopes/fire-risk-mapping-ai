import netCDF4
import os


def get_netcdf_chunk_info(file_path):
    """
    Opens a NetCDF file and prints the chunking information for each variable.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        # Open the dataset in read mode
        with netCDF4.Dataset(file_path, "r") as nc_file:
            print(f"File: {file_path}")
            print("-" * 50)

            # Iterate through all variables in the file
            for var_name, var in nc_file.variables.items():
                # Check if the variable is chunked
                if var.chunking() == "contiguous":
                    chunk_info = "Contiguous (not chunked)"
                else:
                    chunk_info = var.chunking()

                print(f"Variable: {var_name}")
                print(f"  Dimensions: {var.dimensions}")
                print(f"  Chunking: {chunk_info}")
                print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {e}")


# Replace 'path/to/your/IberFire.nc' with the actual path on your local machine
# This script is meant to be run on the client, not in the Dask cluster.
file_path = "/shared-data/IberFire.nc"
# NOTE: The path above is from your cluster, so it will fail.
#       You need to use the path to the file on your local machine
#       or on a machine where you can access the NFS share directly.
#       For example: '/mnt/nfs/shared-data/IberFire.nc'
#
#       A correct path would be:
#       file_path = '/path/to/your/local/copy/of/IberFire.nc'
#
#       The script will not work if the file is only accessible from within your k3s pods.

get_netcdf_chunk_info(file_path)
