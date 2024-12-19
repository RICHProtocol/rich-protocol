import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerAgent(nn.Module):
    def __init__(self, input_dim, num_classes=2, width=512, depth=4, nhead=8):
        """
        Initialize a Transformer-based model with customizable width and depth.
        :param input_dim: Size of input features.
        :param num_classes: Number of output classes.
        :param width: Hidden dimension size of Transformer.
        :param depth: Number of TransformerEncoder layers.
        :param nhead: Number of attention heads.
        """
        super(TransformerAgent, self).__init__()
        self.model_type = 'Transformer'

        self.input_fc = nn.Linear(input_dim, width)  # Initial projection to hidden dimension
        encoder_layer = TransformerEncoderLayer(d_model=width, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(width, num_classes)  # Final classification layer

    def forward(self, src):
        """
        Forward pass for the Transformer model.
        :param src: Input tensor of shape (sequence_length, batch_size, input_dim).
        :return: Output logits.
        """
        src = self.input_fc(src)  # Project input to hidden dimension
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Aggregate across sequence dimension
        return self.fc(output)
