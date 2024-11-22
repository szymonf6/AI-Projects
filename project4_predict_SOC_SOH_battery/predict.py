import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import SOCModel, SOHModel

def capacity_to_soh(capacity, nominal_capacity):
    return 100 * (capacity / nominal_capacity)

def predict_soh():
    # Nominalna pojemność baterii (Ah)
    nominal_capacity = 2.0

    # Wczytanie modelu SOH
    model = SOHModel(input_size=3)
    model.load_state_dict(torch.load('soh_model.pth'))
    model.eval()

    # Wczytanie przygotowanych danych
    data = pd.read_csv('soh_data.csv')
    inputs = torch.tensor(data[['DischargeTime', 'AvgCurrent', 'AvgTemperature']].values, dtype=torch.float32)
    true_capacity = data['Capacity'].values

    # Przewidywanie SOH
    with torch.no_grad():
        predicted_capacity = model(inputs).flatten().numpy()

    # Przeskalowanie wyników do SOH w procentach
    true_soh = capacity_to_soh(true_capacity, nominal_capacity)
    predicted_soh = capacity_to_soh(predicted_capacity, nominal_capacity)

    # Rysowanie wykresu
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(true_soh) + 1), true_soh, label='Prawdziwe SOH (%)', marker='o', color='blue')
    plt.plot(range(1, len(true_soh) + 1), predicted_soh, label='Przewidziane SOH (%)', marker='x', color='orange', linestyle='dashed')
    plt.xlabel('Cykl')
    plt.ylabel('SOH (%)')
    plt.title('Przewidziane SOH dla wszystkich cykli rozładowania')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def voltage_to_soc(voltage, min_voltage, max_voltage):
    return 100 * (voltage - min_voltage) / (max_voltage - min_voltage)

def predict_soc():
    # Parametry skali napięcia
    min_voltage = 2.5  # Minimalne napięcie baterii (w V)
    max_voltage = 4.2  # Maksymalne napięcie baterii (w V)

    # Load model
    input_size = 4
    hidden_size = 128
    output_size = 1
    model = SOCModel(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('soc_model.pth'))
    model.eval()

    # Load data
    df = pd.read_excel('C:/Users/szymo/Desktop/magisterka3semestr/model/dateExcel/ChargeCyclesB0005.xlsx')
    first_discharge_cycle = df[df['Cycle'] == 5]
    first_cycle_data = first_discharge_cycle[['Current_measured', 'Voltage_measured', 'Temperature_measured', 'Time']]
    first_cycle_inputs = torch.tensor(first_cycle_data.values, dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        first_cycle_inputs = first_cycle_inputs.unsqueeze(0)  # Add batch dimension
        predicted_soc = model(first_cycle_inputs)
        predicted_soc = predicted_soc.flatten().numpy()

    # Convert voltage to SOC percentages
    true_soc = voltage_to_soc(first_discharge_cycle['Voltage_measured'].values, min_voltage, max_voltage)
    predicted_soc_percent = voltage_to_soc(predicted_soc, min_voltage, max_voltage)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(first_discharge_cycle['Time'], true_soc, label='Prawdziwe SOC (%)', color='blue')
    plt.plot(first_discharge_cycle['Time'], predicted_soc_percent, label='Przewidziane SOC (%)', color='orange', linestyle='dashed')
    plt.xlabel('Czas, s')
    plt.ylabel('SOC (%)')
    plt.title('Przewidziane SOC dla cyklu rozładowania')
    plt.legend()
    plt.grid()
    plt.show()

def predict_main():
    predict_soc()
    #predict_soh()