# Battery SOC and SOH Estimation

This project leverages deep learning to estimate the **State of Charge (SOC)** and **State of Health (SOH)** of batteries. The goal is to predict these parameters using data from charging and discharging cycles.

## Features
- SOC and SOH prediction using deep learning.
- Model evaluation with multiple metrics (MAE, MSE, RMSE, accuracy).
- Training and prediction scripts with attention mechanisms.

---

## Project Structure

| File/Folder        | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `attention.py`      | Implementation of the Attention mechanism for weighing results in models.  |
| `main.py`           | Main script for training and predicting SOC and SOH.                      |
| `metrics.py`        | Functions to calculate metrics (MAE, MSE, RMSE, accuracy).                |
| `model.py`          | Definitions of SOC and SOH models.                                        |
| `predict.py`        | Script for predicting SOC and SOH using trained models.                   |
| `socModel.py`       | SOC model architecture and implementation.                                |
| `sohModel.py`       | SOH model architecture and implementation.                                |
| `train.py`          | Training scripts for SOC and SOH models.                                  |

---

# Dataset
The model is trained on data from battery charge cycles. The dataset should be in Excel format with the following columns:
- Current_measured: Measured current in the battery (in Amperes).
- Voltage_measured: Measured voltage in the battery (in Volts).
- Temperature_measured: Measured temperature during the charge cycle (in Celsius).
- Time: Time in the charge cycle (in seconds).
- Ensure the correct file path to the dataset is provided in the script or adjust it as needed.

# Model Description
## SOC Prediction Model
The SOC prediction model is a fully connected feed-forward neural network with the following architecture:
- Input Layer: 4 features (current, voltage, temperature, time).
- Hidden Layer: 128 units with ReLU activation.
- Output Layer: 1 output representing the predicted SOC.

## SOH Prediction Model
The SOH prediction model is a linear neural network with three layers and ReLU.

# Training the Model
To train the model, use the provided train_soc_model.py script. This script:

1. Loads the dataset from the specified file.
2. Splits the data into training and testing sets.
3. Trains the model using the Adam optimizer and MSE loss function.
4. Evaluates the model's performance on the test set.
5. Saves the trained model as soc_model.pth.

# Metrics and Visualizations
After training, the following metrics and plots are generated:
1. Metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Accuracy (for classification tasks)
2. Visualizations:
- SOC Predictions vs Actual SOC: Compares predicted and actual SOC values for training and test sets.
- Error Metrics: Plots MSE, RMSE, and MAE for training and test data.
- Accuracy Over Epochs: Shows accuracy evolution during training.