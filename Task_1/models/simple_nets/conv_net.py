import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size, stride, padding)
    )

class ConvNet(nn.Module):
    def __init__(self, num_input_channels, image_size, num_classes):
        super(ConvNet, self).__init__()

        self.block1 = conv_block(num_input_channels, 16)
        self.block2 = conv_block(16, 32)
        self.block3 = conv_block(32, 64)

        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, num_input_channels, image_size, image_size)
            out = self.block1(dummy)
            out = self.block2(out)
            out = self.block3(out)
            flatten_size = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(in_features=flatten_size, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x