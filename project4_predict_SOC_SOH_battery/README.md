# Battery SOC and SOH Estimation

This is a deep learning-based project designed to estimate the State of Charge (SOC) and State of Health (SOH) of a battery. The goal is to predict these parameters based on data from charging and discharging cycles.

## Project Structure

- `attention.py`: Definition of the Attention mechanism, which can be used in models to weigh the results.
- `main.py`: Main file to run the training and prediction process.
- `metrics.py`: Functions to calculate metrics such as MAE, MSE, RMSE, and prediction accuracy.
- `model.py`: Definition of models for SOC (State of Charge) and SOH (State of Health).
- `predict.py`: Code responsible for predicting SOC and SOH values based on trained models.
- `socModel.py`: Definition of the model for SOC estimation.
- `sohModel.py`: Definition of the model for SOH estimation.
- `train.py`: Training code for SOC and SOH models.

## Installation

To run this project, you need to have the following libraries installed:

### Step 1: Clone the repository

```bash
git clone https://github.com/szymonf6/AI-Projects.git
cd AI-Projects

# Dataset

The model is trained on data from battery charge cycles. The data is expected to be in an Excel format with the following columns:

- `Current_measured`: Measured current in the battery (in Amperes)
- `Voltage_measured`: Measured voltage in the battery (in Volts)
- `Temperature_measured`: Measured temperature during the charge cycle (in Celsius)
- `Time`: Time in the charge cycle (in seconds)

Ensure that the Excel file path in the script is correct or adjust it as needed.

## Model Description

The model used for predicting SOC is a fully connected feed-forward neural network with the following architecture:

- **Input Layer**: 4 features (current, voltage, temperature, time)
- **Hidden Layer**: 128 units (ReLU activation)
- **Output Layer**: 1 output representing the predicted voltage (SOC)

## Training the Model

To train the model, run the `train_soc_model.py` script. This script will:

- Load the dataset from the specified file.
- Split the data into training and testing sets.
- Train the model using the Adam optimizer and MSE loss function.
- Evaluate the model's performance on the test set.
- Save the trained model as `soc_model.pth`.

## Run the Training Script

```bash
python train_soc_model.py

