import torch
import torch.nn as nn

class SOHModel(nn.Module):
    def __init__(self, input_size):
        super(SOHModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 8)
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        
        self.linear3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        
        x = self.linear2(x)
        x = self.relu2(x)
        
        x = self.linear3(x)
        return x
