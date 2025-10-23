# Data Preprocessing Scripts

This folder contains a collection of scripts for preprocessing the raw data into a suitable format for training the models.

## Scripts

* `check-correlation.py`: Checks the correlation between features and the target variable.
* `check-nc-chunks.py`: Checks the chunking information of a NetCDF file.
* `check_nc_empty_values.py`: Checks for empty values in a NetCDF file.
* `count_fire.py`: Counts the number of fire and non-fire instances in a CSV file.
* `create-balanced-dataset-csv.py`: Creates a balanced dataset from the raw data.
* `create-train-test-of-old-data.py`: Creates training and testing sets from an older version of the dataset.
* `csv-full-ram.py`: A script to process the entire dataset in RAM.
* `dask-preprocess.py`: A script that uses Dask for preprocessing large datasets.
* `fill_csv_empty_data.py`: Fills missing values in a CSV file.
* `generate-2024.py`: Generates the dataset for the year 2024.
* `generate-balanced-csv.py`: Another script to generate a balanced dataset.
* `generate-csv-dask-paralel.py`: A script that uses Dask for parallel preprocessing.
* `generate-test-train.py`: Generates training and testing sets.
* `open-nc.py`: A script to open and inspect a NetCDF file.
* `slice-dataset-nc-may-to-sep.py`: Slices the dataset to include only the months from May to September.
* `visualize_maps.py`: A script to visualize the generated fire risk maps.

## How to Run

To run any of the scripts, simply use the `python` command followed by the script name, for example:

```bash
python check-correlation.py
