'''
This program predicts emotions from tweets
'''

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np

# Load data from CSV
file_data = pd.read_csv("tweet_emotions.csv")
file_data = file_data.drop(columns="tweet_id")
emotions = file_data['sentiment'].values
sentences = file_data['content'].values

# Remove Twitter usernames from sentences
def remove_usernames(text):
    return re.sub(r'@\w+', '', text)

# Sample function removeextraspaces
def removeextraspaces(sentence):
    return ' '.join(sentence.split())

# Sample function removepunctuation
def removepunctuation(sentence):
    return re.sub(r'[^\w\s]', '', sentence)

# Sample function removelinks
def removelinks(sentence):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sentence)

sentences = np.vectorize(remove_usernames)(sentences)
sentences = np.char.lower(sentences)

# Loop through all elements in the array
for i in range(len(sentences)):
    sentences[i] = removelinks(sentences[i])
    sentences[i] = removeextraspaces(sentences[i])
    sentences[i] = removepunctuation(sentences[i])

emotions = LabelEncoder().fit_transform(emotions)
emotions = torch.tensor(emotions)

class Tokenizer:
    def __init__(self):
        self.word_to_index = {}
        self.next_index = 0

    def tokenize_sentence(self, sentences):
        tokens_list = []
        for sentence in sentences:
            tokens = []
            for word in sentence.split():
                if word not in self.word_to_index:
                    self.word_to_index[word] = self.next_index
                    self.next_index += 1
                tokens.append(self.word_to_index[word])
            tokens_list.append(torch.tensor(tokens))
        return tokens_list

tokenizer = Tokenizer()
sentences = tokenizer.tokenize_sentence(sentences)

# Split data into training and test sets
sentences_train, sentences_test, emotions_train, emotions_test = train_test_split(sentences, emotions, test_size=0.3, random_state=42)

# Convert to tensors and pad to the same length
sentences_train = pad_sequence(sentences_train, batch_first=True)
sentences_test = pad_sequence(sentences_test, batch_first=True)

emotions_train = torch.tensor(emotions_train)
emotions_test = torch.tensor(emotions_test)

# Create DataLoader
train_dataset = TensorDataset(sentences_train, emotions_train)
test_dataset = TensorDataset(sentences_test, emotions_test)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cpu')

class SimpleLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional):
        super(SimpleLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output

# Define model parameters
vocab_size = len(tokenizer.word_to_index)
embedding_dim = 50
hidden_dim = 100
output_dim = 13
num_layers = 2
bidirectional = True

# Create model
model = SimpleLSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions_train = 0
    total_samples_train = 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device).long())
        loss = criterion(outputs, labels.to(device).long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted_train = torch.max(outputs, 1)
        correct_predictions_train += (predicted_train == labels.to(device)).sum().item()
        total_samples_train += labels.size(0)

    average_loss_train = total_loss / len(train_dataloader)
    accuracy_train = correct_predictions_train / total_samples_train

    # Testing the model
    model.eval()
    correct_predictions_test = 0
    total_samples_test = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs_test = model(inputs.to(device).long())
            _, predicted_test = torch.max(outputs_test, 1)
            correct_predictions_test += (predicted_test == labels.to(device)).sum().item()
            total_samples_test += labels.size(0)

    accuracy_test = correct_predictions_test / total_samples_test

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {average_loss_train:.4f}, Train Accuracy: {accuracy_train:.4f}, '
          f'Test Accuracy: {accuracy_test:.4f}')
