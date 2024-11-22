import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import SOCModel, SOHModel
from metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error, calculate_accuracy

def voltage_to_soc(voltage, min_voltage, max_voltage):
    return 100 * (voltage - min_voltage) / (max_voltage - min_voltage)

def train_soc_model():
    # Load data from file
    df = pd.read_excel('ChargeCyclesB0005.xlsx')

    # Select columns from the Excel file
    soc_data = df[['Current_measured', 'Voltage_measured', 'Temperature_measured', 'Time']]
    soc_labels = df['Voltage_measured']  # SOC

    # Convert Excel data to tensors
    soc_inputs = torch.tensor(soc_data.values, dtype=torch.float32)
    soc_labels = torch.tensor(soc_labels.values, dtype=torch.float32).unsqueeze(1)

    # Create SOC dataset
    soc_dataset = TensorDataset(soc_inputs, soc_labels)
    soc_train_size = int(0.7 * len(soc_dataset))
    soc_test_size = len(soc_dataset) - soc_train_size
    soc_train_dataset, soc_test_dataset = random_split(soc_dataset, [soc_train_size, soc_test_size])

    batch_size = 128
    soc_train_dataloader = DataLoader(soc_train_dataset, batch_size=batch_size, shuffle=True)
    soc_test_dataloader = DataLoader(soc_test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    input_size = 4
    hidden_size = 128
    output_size = 1

    # Initialize model
    model = SOCModel(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Parametry skali napięcia
    min_voltage = 3.0  # Minimalne napięcie baterii (w V)
    max_voltage = 4.2  # Maksymalne napięcie baterii (w V)

    # Number of training epochs
    num_epochs = 10

    train_losses = {'mse': [], 'rmse': [], 'mae': []}
    test_losses = {'mse': [], 'rmse': [], 'mae': []}
    train_accuracies = []
    test_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_metrics = {'mse': [], 'rmse': [], 'mae': [], 'accuracy': []}

        # Training SOC
        for batch in soc_train_dataloader:
            soc_inputs_batch, soc_labels_batch = batch

            optimizer.zero_grad()
            soc_output = model(soc_inputs_batch)
            soc_loss = criterion(soc_output, soc_labels_batch)
            soc_loss.backward()
            optimizer.step()

            train_metrics['mse'].append(mean_squared_error(soc_output, soc_labels_batch))
            train_metrics['rmse'].append(root_mean_squared_error(soc_output, soc_labels_batch))
            train_metrics['mae'].append(mean_absolute_error(soc_output, soc_labels_batch))
            train_metrics['accuracy'].append(calculate_accuracy(soc_output, soc_labels_batch))
        
        # Średnie wartości błędów dla epoki treningowej
        train_losses['mse'].append(np.mean(train_metrics['mse']))
        train_losses['rmse'].append(np.mean(train_metrics['rmse']))
        train_losses['mae'].append(np.mean(train_metrics['mae']))
        train_accuracies.append(np.mean(train_metrics['accuracy']))

        print(f'Epoch [{epoch+1}/{num_epochs}], Train SOC MSE: {train_losses["mse"][-1]:.4f}')
        print(f'Train SOC RMSE: {train_losses["rmse"][-1]:.4f}')
        print(f'Train SOC MAE: {train_losses["mae"][-1]:.4f}')
        print(f'Train SOC Accuracy: {train_accuracies[-1]:.4f}')

        # Evaluation
        model.eval()
        test_metrics = {'mse': [], 'rmse': [], 'mae': [], 'accuracy': []}
        true_voltages = []
        predicted_voltages = []

        with torch.no_grad():
            for batch in soc_test_dataloader:
                soc_inputs_batch, soc_labels_batch = batch
                soc_output = model(soc_inputs_batch)
                test_metrics['mse'].append(mean_squared_error(soc_output, soc_labels_batch))
                test_metrics['rmse'].append(root_mean_squared_error(soc_output, soc_labels_batch))
                test_metrics['mae'].append(mean_absolute_error(soc_output, soc_labels_batch))
                test_metrics['accuracy'].append(calculate_accuracy(soc_output, soc_labels_batch))
                true_voltages.extend(soc_labels_batch.numpy())
                predicted_voltages.extend(soc_output.numpy())

        # Średnie wartości błędów dla epoki testowej
        test_losses['mse'].append(np.mean(test_metrics['mse']))
        test_losses['rmse'].append(np.mean(test_metrics['rmse']))
        test_losses['mae'].append(np.mean(test_metrics['mae']))
        test_accuracies.append(np.mean(test_metrics['accuracy']))

        # Przeskalowanie wyników do SOC
        true_soc = voltage_to_soc(np.array(true_voltages), min_voltage, max_voltage)
        predicted_soc = voltage_to_soc(np.array(predicted_voltages), min_voltage, max_voltage)

        print(f'Epoch [{epoch+1}/{num_epochs}], Test SOC MSE: {test_losses["mse"][-1]:.4f}')
        print(f'Test SOC RMSE: {test_losses["rmse"][-1]:.4f}')
        print(f'Test SOC MAE: {test_losses["mae"][-1]:.4f}')
        print(f'Test SOC Accuracy: {test_accuracies[-1]:.4f}')

    torch.save(model.state_dict(), 'soc_model.pth')

    # Rysowanie wykresów
    epochs = range(1, num_epochs + 1)
    '''
    
    # Wykres przewidywanych SOC i prawdziwych SOC
    plt.figure(figsize=(10,5))
    plt.figure(0)
    plt.plot(true_soc, label='Prawdziwe SOC (%)')
    plt.plot(predicted_soc, label='Przewidywane SOC (%)', linestyle='dashed')
    plt.xlabel('Epoka')
    plt.ylabel('SOC (%)')
    plt.legend()
    plt.title('Prawdziwe vs Przewidywane SOC')
    plt.show()

    plt.figure(1)
    plt.plot(epochs, train_losses['mse'], label='MSE treningowe')
    plt.plot(epochs, test_losses['mse'], label='MSE testowe')
    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.title('Średni błąd kwardatowy MSE')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.plot(epochs, train_losses['rmse'], label='RMSE treningowe')
    plt.plot(epochs, test_losses['rmse'], label='RMSE testowe')
    plt.xlabel('Epoka')
    plt.ylabel('RMSE')
    plt.title('Pierwiastek ze średniego błędu kwadratowego RMSE')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(3)
    plt.plot(epochs, train_losses['mae'], label='MAE treningowe')
    plt.plot(epochs, test_losses['mae'], label='MAE testowe')
    plt.xlabel('Epoka')
    plt.ylabel('MAE')
    plt.title('Błąd bezwzględny MAE')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(4)
    plt.plot(epochs, train_accuracies, label='Dokładność treningowa')
    plt.plot(epochs, test_accuracies, label='Dokładność testowa')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność, %')
    plt.title('Dokładność w zależności od epoki')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Wykresy błędów w procentach
    plt.figure(5)
    plt.plot(epochs, np.array(train_losses['mse']) * 100 / max_voltage, label='MSE w % (treningowe)')
    plt.plot(epochs, np.array(test_losses['mse']) * 100 / max_voltage, label='MSE w % (testowe)')
    plt.xlabel('Epoka')
    plt.ylabel('MSE (%)')
    plt.title('Średni błąd kwadratowy MSE w procentach')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(6)
    plt.plot(epochs, np.array(train_losses['rmse']) * 100 / max_voltage, label='RMSE w % (treningowe)')
    plt.plot(epochs, np.array(test_losses['rmse']) * 100 / max_voltage, label='RMSE w % (testowe)')
    plt.xlabel('Epoka')
    plt.ylabel('RMSE (%)')
    plt.title('Pierwiastek ze średniego błędu kwadratowego RMSE w procentach')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(7)
    plt.plot(epochs, np.array(train_losses['mae']) * 100 / max_voltage, label='MAE w % (treningowe)')
    plt.plot(epochs, np.array(test_losses['mae']) * 100 / max_voltage, label='MAE w % (testowe)')
    plt.xlabel('Epoka')
    plt.ylabel('MAE (%)')
    plt.title('Błąd bezwzględny MAE w procentach')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    '''

    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_accuracies, label='Dokładność treningowa')
    plt.plot(epochs, test_accuracies, label='Dokładność testowa')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność, %')
    plt.title('Dokładność w zależności od epoki')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Wykres MAE
    fig, ax1 = plt.subplots()

    ln1 = ax1.plot(epochs, train_losses['mae'], label='MAE (treningowe)', color='blue')
    ln2 = ax1.plot(epochs, test_losses['mae'], label='MAE (testowe)', color='orange')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('MAE', color='blue')
    #ax1.set_ylim(0, 1)
    ax1.grid()

    ax2 = ax1.twinx()
    ln3 = ax2.plot(epochs, np.array(train_losses['mae']) * 100 / max_voltage, label='MAE w % (treningowe)', color='blue', linestyle='dashed')
    ln4 = ax2.plot(epochs, np.array(test_losses['mae']) * 100 / max_voltage, label='MAE w % (testowe)', color='orange', linestyle='dashed')
    ax2.set_ylabel('MAE (%)', color='orange')
    #ax2.set_ylim(0, 100)

    # Łączenie legend
    lns = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper right')

    plt.title('Średni błąd bezwzględny (MAE) i MAE w %')
    plt.tight_layout()
    plt.show()

    # Wykres MSE
    fig, ax1 = plt.subplots()

    ln1 = ax1.plot(epochs, train_losses['mse'], label='MSE (treningowe)', color='blue')
    ln2 = ax1.plot(epochs, test_losses['mse'], label='MSE (testowe)', color='orange')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('MSE', color='blue')
    #ax1.set_ylim(0, 1)
    ax1.grid()

    ax2 = ax1.twinx()
    ln3 = ax2.plot(epochs, np.array(train_losses['mse']) * 100 / max_voltage, label='MSE w % (treningowe)', color='blue', linestyle='dashed')
    ln4 = ax2.plot(epochs, np.array(test_losses['mse']) * 100 / max_voltage, label='MSE w % (testowe)', color='orange', linestyle='dashed')
    ax2.set_ylabel('MSE (%)', color='orange')
    #ax2.set_ylim(0, 100)

    # Łączenie legend
    lns = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper right')

    plt.title('Średni błąd kwadratowy (MSE) i MSE w %')
    plt.tight_layout()
    plt.show()

    # Wykres MSE
    fig, ax1 = plt.subplots()

    ln1 = ax1.plot(epochs, train_losses['rmse'], label='RMSE (treningowe)', color='blue')
    ln2 = ax1.plot(epochs, test_losses['rmse'], label='RMSE (testowe)', color='orange')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('RMSE', color='blue')
    #ax1.set_ylim(0, 1)
    ax1.grid()

    ax2 = ax1.twinx()
    ln3 = ax2.plot(epochs, np.array(train_losses['rmse']) * 100 / max_voltage, label='RMSE w % (treningowe)', color='blue', linestyle='dashed')
    ln4 = ax2.plot(epochs, np.array(test_losses['rmse']) * 100 / max_voltage, label='RMSE w % (testowe)', color='orange', linestyle='dashed')
    ax2.set_ylabel('RMSE (%)', color='orange')
    #ax2.set_ylim(0, 100)

    # Łączenie legend
    lns = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper right')

    plt.title('Pierwiastek ze średniego błędu kwadratowego (RMSE) i RMSE w %')
    plt.tight_layout()
    plt.show()
    
def capacity_to_soh(capacity, nominal_capacity):
    return 100 * (capacity / nominal_capacity)

def train_soh_model():
    # Wczytanie przygotowanych danych
    data = pd.read_csv('soh_data.csv')
    inputs = data[['DischargeTime', 'AvgCurrent', 'AvgTemperature']].values
    labels = data['Capacity'].values

    # Nominalna pojemność baterii (Ah)
    nominal_capacity = 2.0

    # Konwersja do tensorów
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Tworzenie zestawu danych
    dataset = TensorDataset(inputs, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Tworzenie DataLoaderów
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Inicjalizacja modelu, funkcji straty i optymalizatora
    model = SOHModel(input_size=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Parametry treningowe
    num_epochs = 20

    train_losses = {'mse': [], 'rmse': [], 'mae': []}
    test_losses = {'mse': [], 'rmse': [], 'mae': []}
    train_accuracies = []
    test_accuracies = []

    # Trening modelu
    for epoch in range(num_epochs):
        model.train()
        train_metrics = {'mse': [], 'rmse': [], 'mae': [], 'accuracy': []}

        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_metrics['mse'].append(mean_squared_error(outputs, batch_labels))
            train_metrics['rmse'].append(root_mean_squared_error(outputs, batch_labels))
            train_metrics['mae'].append(mean_absolute_error(outputs, batch_labels))
            train_metrics['accuracy'].append(calculate_accuracy(outputs, batch_labels))

        train_losses['mse'].append(np.mean(train_metrics['mse']))
        train_losses['rmse'].append(np.mean(train_metrics['rmse']))
        train_losses['mae'].append(np.mean(train_metrics['mae']))
        train_accuracies.append(np.mean(train_metrics['accuracy']))

        print(f'Epoch [{epoch+1}/{num_epochs}], Train SOH MSE: {np.mean(train_metrics["mse"]):.4f}')
        print(f'Train SOH RMSE: {np.mean(train_metrics["rmse"]):.4f}')
        print(f'Train SOH MAE: {np.mean(train_metrics["mae"]):.4f}')
        print(f'Train SOH Accuracy: {train_accuracies[-1]:.4f}')

        # Testowanie modelu
        model.eval()
        test_metrics = {'mse': [], 'rmse': [], 'mae': [], 'accuracy': []}

        with torch.no_grad():
            for batch_inputs, batch_labels in test_loader:
                outputs = model(batch_inputs)
                test_metrics['mse'].append(mean_squared_error(outputs, batch_labels))
                test_metrics['rmse'].append(root_mean_squared_error(outputs, batch_labels))
                test_metrics['mae'].append(mean_absolute_error(outputs, batch_labels))
                test_metrics['accuracy'].append(calculate_accuracy(outputs, batch_labels))

        test_losses['mse'].append(np.mean(test_metrics['mse']))
        test_losses['rmse'].append(np.mean(test_metrics['rmse']))
        test_losses['mae'].append(np.mean(test_metrics['mae']))
        test_accuracies.append(np.mean(test_metrics['accuracy']))

        print(f'Epoch [{epoch+1}/{num_epochs}], Test SOH MSE: {np.mean(test_metrics["mse"]):.4f}')
        print(f'Test SOH RMSE: {np.mean(test_metrics["rmse"]):.4f}')
        print(f'Test SOH MAE: {np.mean(test_metrics["mae"]):.4f}')
        print(f'Test SOH Accuracy: {test_accuracies[-1]:.4f}')

    torch.save(model.state_dict(), 'soh_model.pth')

    # Rysowanie wykresów
    epochs = range(1, num_epochs + 1)
    '''
    
    plt.figure(figsize=(10,5))
    plt.figure(0)
    plt.plot(epochs, train_losses['mse'], label='MSE treningowe')
    plt.plot(epochs, test_losses['mse'], label='MSE testowe')
    plt.ylim(0, 1)
    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.title('Średni błąd kwadratowy MSE')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(1)
    plt.plot(epochs, train_losses['rmse'], label='RMSE treningowe')
    plt.plot(epochs, test_losses['rmse'], label='RMSE testowe')
    plt.ylim(0, 1)
    plt.xlabel('Epoka')
    plt.ylabel('RMSE')
    plt.title('Pierwiastek ze średniego błędu kwadratowego RMSE')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.plot(epochs, train_losses['mae'], label='MAE treningowe')
    plt.plot(epochs, test_losses['mae'], label='MAE testowe')
    plt.ylim(0, 1)
    plt.xlabel('Epoka')
    plt.ylabel('MAE')
    plt.title('Błąd bezwzględny MAE')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(3)
    plt.plot(epochs, train_accuracies, label='Dokładność treningowa')
    plt.plot(epochs, test_accuracies, label='Dokładność testowa')

    plt.xlabel('Epoka')
    plt.ylabel('Dokładność, %')
    plt.title('Dokładność w zależności od epoki')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Wykresy błędów w procentach
    plt.figure(5)
    plt.plot(epochs, np.array(train_losses['mse']) * 100 / nominal_capacity, label='MSE w % (treningowe)')
    plt.plot(epochs, np.array(test_losses['mse']) * 100 / nominal_capacity, label='MSE w % (testowe)')
    plt.ylim(0, 10)
    plt.xlabel('Epoka')
    plt.ylabel('MSE (%)')
    plt.title('Średni błąd kwadratowy MSE w procentach')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(6)
    plt.plot(epochs, np.array(train_losses['rmse']) * 100 / nominal_capacity, label='RMSE w % (treningowe)')
    plt.plot(epochs, np.array(test_losses['rmse']) * 100 / nominal_capacity, label='RMSE w % (testowe)')
    plt.ylim(0, 10)
    plt.xlabel('Epoka')
    plt.ylabel('RMSE (%)')
    plt.title('Pierwiastek ze średniego błędu kwadratowego RMSE w procentach')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(7)
    plt.plot(epochs, np.array(train_losses['mae']) * 100 / nominal_capacity, label='MAE w % (treningowe)')
    plt.plot(epochs, np.array(test_losses['mae']) * 100 / nominal_capacity, label='MAE w % (testowe)')
    plt.ylim(0, 10)
    plt.xlabel('Epoka')
    plt.ylabel('MAE (%)')
    plt.title('Błąd bezwzględny MAE w procentach')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    '''
    

    # Wykres MAE
    fig, ax1 = plt.subplots()

    ln1 = ax1.plot(epochs, train_losses['mae'], label='MAE (treningowe)', color='blue')
    ln2 = ax1.plot(epochs, test_losses['mae'], label='MAE (testowe)', color='orange')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('MAE', color='blue')
    ax1.set_ylim(0, 1)
    ax1.grid()

    ax2 = ax1.twinx()
    ln3 = ax2.plot(epochs, np.array(train_losses['mae']) * 100 / nominal_capacity, label='MAE w % (treningowe)', color='blue', linestyle='dashed')
    ln4 = ax2.plot(epochs, np.array(test_losses['mae']) * 100 / nominal_capacity, label='MAE w % (testowe)', color='orange', linestyle='dashed')
    ax2.set_ylabel('MAE (%)', color='orange')
    ax2.set_ylim(0, 100)

    # Łączenie legend
    lns = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper right')

    plt.title('Średni błąd bezwzględny (MAE) i MAE w %')
    plt.tight_layout()
    plt.show()

    # Wykres MSE
    fig, ax1 = plt.subplots()

    ln1 = ax1.plot(epochs, train_losses['mse'], label='MSE (treningowe)', color='blue')
    ln2 = ax1.plot(epochs, test_losses['mse'], label='MSE (testowe)', color='orange')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('MSE', color='blue')
    ax1.set_ylim(0, 1)
    ax1.grid()

    ax2 = ax1.twinx()
    ln3 = ax2.plot(epochs, np.array(train_losses['mse']) * 100 / nominal_capacity, label='MSE w % (treningowe)', color='blue', linestyle='dashed')
    ln4 = ax2.plot(epochs, np.array(test_losses['mse']) * 100 / nominal_capacity, label='MSE w % (testowe)', color='orange', linestyle='dashed')
    ax2.set_ylabel('MSE (%)', color='orange')
    ax2.set_ylim(0, 100)

    # Łączenie legend
    lns = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper right')

    plt.title('Średni błąd kwadratowy (MSE) i MSE w %')
    plt.tight_layout()
    plt.show()

    # Wykres MSE
    fig, ax1 = plt.subplots()

    ln1 = ax1.plot(epochs, train_losses['rmse'], label='RMSE (treningowe)', color='blue')
    ln2 = ax1.plot(epochs, test_losses['rmse'], label='RMSE (testowe)', color='orange')
    ax1.set_xlabel('Epoka')
    ax1.set_ylabel('RMSE', color='blue')
    ax1.set_ylim(0, 1)
    ax1.grid()

    ax2 = ax1.twinx()
    ln3 = ax2.plot(epochs, np.array(train_losses['rmse']) * 100 / nominal_capacity, label='RMSE w % (treningowe)', color='blue', linestyle='dashed')
    ln4 = ax2.plot(epochs, np.array(test_losses['rmse']) * 100 / nominal_capacity, label='RMSE w % (testowe)', color='orange', linestyle='dashed')
    ax2.set_ylabel('RMSE (%)', color='orange')
    ax2.set_ylim(0, 100)

    # Łączenie legend
    lns = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper right')

    plt.title('Pierwiastek ze średniego błędu kwadratowego (RMSE) i RMSE w %')
    plt.tight_layout()
    plt.show()

def train_main():
    train_soc_model()
    #train_soh_model()