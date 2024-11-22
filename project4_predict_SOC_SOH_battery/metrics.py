import torch

def mean_absolute_error(predictions, labels):
    return torch.mean(torch.abs(predictions - labels)).item()

def mean_squared_error(predictions, labels):
    return torch.mean((predictions - labels) ** 2).item()

def root_mean_squared_error(predictions, labels):
    return torch.sqrt(torch.mean((predictions - labels) ** 2)).item()

def calculate_accuracy(output, target, tolerance=0.05):
    correct = ((torch.abs(output - target) < tolerance).float()).sum().item()
    total = target.size(0)
    accuracy = correct / total * 100
    return accuracy
