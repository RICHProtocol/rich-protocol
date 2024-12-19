import torch
import torch.nn as nn

class ShuffleNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        super(ShuffleNetBlock, self).__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dwconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dwconv(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return self.relu(out)

class ShuffleNetAgent(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, width=64, depth=3):
        """
        Initialize ShuffleNet with group convolutions.
        :param input_channels: Number of input channels.
        :param num_classes: Number of output classes.
        :param width: Group width for convolution layers.
        :param depth: Number of ShuffleNet blocks.
        """
        super(ShuffleNetAgent, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.blocks = self._make_layer(width, width * 2, depth)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 2, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(ShuffleNetBlock(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
