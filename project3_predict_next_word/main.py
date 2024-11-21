'''
This program predicts next word 
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

with open('data.txt', 'r') as f:
    text = f.read()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# convert text to integers
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

# encode the text
encoded = np.array([char2int[ch] for ch in text])

def one_hot_encode(arr, n_labels):
    # initialize the encode array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

def get_batches(arr, batch_size, seq_length):
    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr) // batch_size_total

    # keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # the features
        x = arr[:, n:n+seq_length]
        # the targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

def plot_training_curve(train_losses, val_losses, accuracies):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='orange')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(accuracies, label='Val Accuracy', color='green')
    ax2.legend(loc='upper right')

    plt.show()

def calculate_accuracy(net, data, batch_size, seq_length):
    net.eval()
    correct_predictions = 0
    total_predictions = 0

    h = net.init_hidden(batch_size)

    for x, y in get_batches(data, batch_size, seq_length):
        x = one_hot_encode(x, len(net.chars))
        inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

        h = tuple([each.data.to(device) for each in h])

        output, h = net(inputs, h)

        predictions = torch.argmax(output, dim=1)
        targets = targets.view(batch_size * seq_length).long()

        correct_predictions += (predictions == targets).sum().item()
        total_predictions += targets.numel()

    accuracy = correct_predictions / total_predictions * 100
    return accuracy

class Model(nn.Module):
    def __init__(self, tokens, n_hidden=32, n_layers=1, drop_prob=0.5, lr=0.01):
        super(Model, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # Using bidirectional LSTM with multiple layers
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)

        # Additional LSTM layers
        self.lstm_layers = nn.ModuleList([nn.LSTM(n_hidden * 2, n_hidden, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True) for _ in range(2)])

        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden * 2, len(self.chars))

    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)

        # Pass through additional LSTM layers
        for layer in self.lstm_layers:
            r_output, hidden = layer(r_output, hidden)

        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden * 2)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers * 2, batch_size, self.n_hidden).zero_())
        return hidden

def evaluate(net, data, batch_size, seq_length, loss_fn):
    net.eval()
    h = net.init_hidden(batch_size)
    losses = []
    correct_preds = 0
    total_preds = 0

    for x, y in get_batches(data, batch_size, seq_length):
        x = one_hot_encode(x, len(net.chars))
        inputs, targets = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

        h = tuple([each.data.to(device) for each in h])
        output, h = net(inputs, h)
        
        loss = loss_fn(output, targets.view(batch_size * seq_length).long())
        losses.append(loss.item())

        predictions = torch.argmax(output, dim=1)
        correct_preds += (predictions == targets.view(-1)).sum().item()
        total_preds += targets.numel()

    accuracy = (correct_preds / total_preds) * 100
    return np.mean(losses), accuracy

# Training function
def train(net, data, epochs=5, batch_size=10, seq_length=25, lr=0.01, clip=5, val_frac=0.1, print_every=10):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Split data into training and validation
    val_idx = int(len(data) * (1 - val_frac))
    train_data, val_data = data[:val_idx], data[val_idx:]

    counter = 0
    n_chars = len(net.chars)
    for epoch in range(epochs):
        h = net.init_hidden(batch_size)

        for x, y in get_batches(train_data, batch_size, seq_length):
            counter += 1

            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

            h = tuple([each.data.to(device) for each in h])

            net.zero_grad()

            output, h = net(inputs, h)
            loss = loss_fn(output, targets.view(batch_size * seq_length).long())

            loss.backward()

            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # Validation
            if counter % print_every == 0:
                val_loss, val_accuracy = evaluate(net, val_data, batch_size, seq_length, loss_fn)
                train_losses.append(loss.item())
                val_losses.append(val_loss)

                print(f"Epoch: {epoch+1}/{epochs}... Step: {counter}... "
                      f"Loss: {loss.item():.4f}... Val Loss: {val_loss:.4f}... Accuracy: {val_accuracy:.2f}%")

# define and print the net
n_hidden = 32  
n_layers = 1    

train_losses, val_losses = [], []

net = Model(chars, n_hidden, n_layers)

batch_size = 128
seq_length = 100
epochs = 10
train_losses, val_losses = [], []

# train the model
train(net, encoded, epochs=epochs, batch_size=batch_size, seq_length=seq_length, lr=0.01, clip=5, val_frac=0.1, print_every=10)

model_name = 'model.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}
with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)

def predict(net, char, h=None, top_k=None):
    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x).to(device)

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)
    # get the character probabilities
    p = F.softmax(out, dim=1).data

    # get top characters
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.cpu().numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.cpu().numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h

def sample(net, size, prime, top_k=None):

    net.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)
    return ''.join(chars)

plot_training_curve(train_losses, val_losses)
plt.show()