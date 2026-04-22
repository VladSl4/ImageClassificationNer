import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from interfaces.mnist_interface import MnistClassifierInterface
from models.simple_nets.conv_net import ConvNet


class ConvolutionalClassifier(MnistClassifierInterface):
    def __init__(self, num_input_channels=1, image_size=28, num_classes=10, lr=0.001, batch_size=32, num_epochs=5):
        self.num_input_channels = num_input_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvNet(
            num_input_channels=self.num_input_channels,
            image_size=self.image_size,
            num_classes=self.num_classes
        )

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, x_train, y_train):
        x_train_reshaped = x_train.reshape(-1, self.num_input_channels, self.image_size, self.image_size)

        tensor_x = torch.tensor(x_train_reshaped, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.int64)

        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader_train = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            num_processed = 0
            for features, labels in dataloader_train:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * features.size(0)
                num_processed += len(labels)
            print(f'CNN Epoch {epoch}, loss: {running_loss / num_processed}')

    def predict(self, x_test):
        self.model.eval()

        x_test_reshaped = x_test.reshape(-1, self.num_input_channels, self.image_size, self.image_size)
        tensor_x = torch.tensor(x_test_reshaped, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        dataloader_test = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        predictions = []
        with torch.no_grad():
            for (features,) in dataloader_test:
                features = features.to(self.device)

                outputs = self.model(features)
                cat = torch.argmax(outputs, dim=-1)
                predictions.extend(cat.cpu().tolist())

        return np.array(predictions)
