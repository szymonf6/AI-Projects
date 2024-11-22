import torch
import torch.nn as nn

class SOCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SOCModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        soc_output = self.fc(lstm_out)
        return soc_output
    
class SOHModel(nn.Module):
    def __init__(self, input_size):
        super(SOHModel, self).__init__()
        #pierwsza warstwa
        self.linear1 = nn.Linear(input_size, 8)
        self.relu1 = nn.ReLU()
        
        #druga warstwa
        self.linear2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        
        #trzecia warstwa
        self.linear3 = nn.Linear(8, 1)

    def forward(self, x):
        #pierwsza warstwa
        x = self.linear1(x)
        x = self.relu1(x)
        
        #druga warstwa
        x = self.linear2(x)
        x = self.relu2(x)

        #trzecia warstwa
        x = self.linear3(x)
        return x