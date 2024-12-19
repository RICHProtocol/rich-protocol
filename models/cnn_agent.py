import torch
import torch.nn as nn

class CNNAgent(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, width=16, depth=3):
        """
        Initialize a CNN model with customizable width and depth.
        :param input_channels: Number of input channels (e.g., 3 for RGB).
        :param num_classes: Number of output classes.
        :param width: Number of filters in convolutional layers.
        :param depth: Number of convolutional layers.
        """
        super(CNNAgent, self).__init__()
        self.depth = depth
        self.width = width

        # Dynamically create convolutional layers
        self.conv_layers = self._make_layer(input_channels, width, depth)
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(width * 8 * 8, 128),  # Assuming input image size of 32x32
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, input_channels, width, depth):
        """
        Create convolutional layers dynamically.
        :param input_channels: Number of input channels.
        :param width: Number of filters per layer.
        :param depth: Number of convolutional layers.
        """
        layers = []
        in_channels = input_channels
        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = width  # Output channels become the input channels for the next layer
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the CNN model.
        :param x: Input tensor of shape (batch_size, channels, height, width).
        :return: Output logits.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        return self.fc(x)

