# Deep CNN + LightGBM Hybrid Models

This folder contains hybrid models that combine a deep CNN with a LightGBM model for wildfire prediction. It includes three different ensembling techniques:

* **Averaging**: Simple averaging of the predictions from both models.
* **Stacking**: A meta-model is trained on the predictions of the base models.
* **Weighted Averaging**: A weighted average of the predictions from both models, with the weights being optimized.

## How to Run

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the desired script**:
    * For the averaging model:
        ```bash
        python cnn-lightgbm-avrg.py
        ```
    * For the stacking model:
        ```bash
        python cnn-lightgbm-stacking.py
        ```
    * For the weighted model:
        ```bash
        python cnn-lightgbm-weight.py
        ```
