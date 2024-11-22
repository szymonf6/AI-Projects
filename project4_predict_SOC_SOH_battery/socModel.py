import torch
import torch.nn as nn

class SOCModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SOCModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        lstm_out, _ = self.lstm(rnn_out)
        lstm_out = lstm_out[:, -1, :]
        soc_output = self.fc(lstm_out)
        return soc_output