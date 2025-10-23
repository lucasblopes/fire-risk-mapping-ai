# Wildfire Prediction using Deep Learning and Machine Learning

This repository contains a collection of scripts and models for predicting wildfire risk using various machine learning and deep learning techniques. The project is structured into several folders, each containing a different model or a set of related scripts.

## Project Structure

* **/convnext**: Contains a wildfire prediction model based on the ConvNeXtV2 architecture.
* **/deep-cnn**: Implements a deep convolutional neural network (CNN) for wildfire prediction.
* **/deep-cnn-lightgbm**: Contains hybrid models that combine a deep CNN with a LightGBM model using different ensembling techniques (averaging, stacking, and weighted averaging).
* **/machine-learning**: Includes implementations of traditional machine learning models such as RandomForest, XGBoost, and LightGBM.
* **/metrics**: This folder stores the performance metrics of the different models in `.txt` files.
* **/preprocess-csv**: A collection of scripts for preprocessing the raw data into a suitable format for training the models. This includes scripts for creating balanced datasets, handling missing values, and generating features.
* **/quantum-cnn**: An experimental model that uses a quantum convolutional neural network (QuantumCNN) for wildfire prediction.
* **/resnet**: Implements a wildfire prediction model based on the ResNet architecture.
* **/simple-cnn**: A basic implementation of a convolutional neural network.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Navigate to a model's folder:**
    ```bash
    cd <model-folder>
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the main script:**
    ```bash
    python <script-name>.py
    ```

Please refer to the README file in each subfolder for more detailed instructions on how to run the specific models and scripts.
