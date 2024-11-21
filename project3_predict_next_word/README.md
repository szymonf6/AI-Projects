# Next Word Prediction

This is a PyTorch-based program for next-word prediction. The program uses a bidirectional LSTM neural network to learn and predict the next character in a given sequence of text.

---

## Features
- **Bidirectional LSTM**: Utilizes a bidirectional LSTM architecture with multiple layers for more accurate predictions.
- **Custom Tokenization**: Converts text to numerical representations and one-hot encodes the data for training.
- **Dynamic Sampling**: Predicts the next character based on probabilities, allowing diverse outputs.
- **Training and Validation**: Includes training loops, validation, and accuracy calculation.
- **Visualization**: Plots training and validation losses to monitor performance.

---

## Prerequisites
To run this project, you need Python 3.8+ and the following dependencies installed:

- `numpy`
- `torch`
- `matplotlib`

You can install the dependencies using the command:
```bash
pip install -r requirements.txt
