import torch
import torch.nn as nn

class LSTMAgent(nn.Module):
    def __init__(self, input_dim, hidden_width=64, output_dim=2, depth=2):
        """
        Initialize an LSTM model with customizable depth and width.
        :param input_dim: Number of input features.
        :param hidden_width: Number of hidden units in LSTM layers.
        :param output_dim: Number of output classes.
        :param depth: Number of LSTM layers.
        """
        super(LSTMAgent, self).__init__()
        self.depth = depth
        self.hidden_width = hidden_width

        # LSTM layers
        self.lstm = self._make_layer(input_dim, hidden_width, depth)
        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_width, output_dim)

    def _make_layer(self, input_dim, hidden_width, depth):
        """
        Create an LSTM layer stack dynamically.
        :param input_dim: Number of input features.
        :param hidden_width: Number of hidden units.
        :param depth: Number of LSTM layers.
        """
        return nn.LSTM(input_dim, hidden_width, num_layers=depth, batch_first=True)

    def forward(self, x):
        """
        Forward pass for the LSTM model.
        :param x: Input tensor of shape (batch_size, sequence_length, input_dim).
        :return: Output logits.
        """
        _, (hidden_state, _) = self.lstm(x)
        output = self.fc(hidden_state[-1])  # Use the last layer's hidden state
        return output
