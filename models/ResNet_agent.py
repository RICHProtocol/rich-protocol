import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """A basic residual block for ResNet."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ResNetAgent(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, width=64, depth=3):
        """
        Initialize a ResNet-like model with configurable width and depth.
        :param input_channels: Number of input channels.
        :param num_classes: Number of output classes.
        :param width: Number of channels in the residual blocks.
        :param depth: Number of residual blocks per layer.
        """
        super(ResNetAgent, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, width, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(width, width, depth)
        self.layer2 = self._make_layer(width, width * 2, depth, stride=2)
        self.layer3 = self._make_layer(width * 2, width * 4, depth, stride=2)
        self.layer4 = self._make_layer(width * 4, width * 8, depth, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width * 8, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Create residual blocks dynamically.
        :param in_channels: Input channels.
        :param out_channels: Output channels.
        :param blocks: Number of residual blocks.
        :param stride: Stride size.
        """
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
