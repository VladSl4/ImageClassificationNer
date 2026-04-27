import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms, datasets

from interfaces.cv_interface import ImageClassifierInterface


class AnimalClassifier(ImageClassifierInterface):
    def __init__(self, weights_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = weights_path
        self.class_names = []

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if self.weights_path and os.path.exists(self.weights_path):
            self._load_weights()

    def _load_weights(self):
        checkpoint = torch.load(self.weights_path, map_location=self.device)

        self.class_names = checkpoint['class_names']

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def train(self, dataset_path: str, **kwargs):
        batch_size = kwargs.get('batch_size', 32)
        num_epochs = kwargs.get('num_epochs', 5)
        learning_rate = kwargs.get('learning_rate', 0.001)
        save_path = kwargs.get('save_path', 'weights/best_cv_model.pth')

        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(dataset_path, train_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.class_names = dataset.classes

        print(f'Found {len(self.class_names)} classes: {self.class_names}')

        for param in self.model.parameters():
            param.requires_grad = False

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(self.class_names))
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataset)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        checkpoint = {
            'state_dict': self.model.state_dict(),
            'class_names': self.class_names
        }
        torch.save(checkpoint, save_path)
        print(f'Model saved to {save_path}')

    def predict(self, image_path: str, **kwargs) -> str:
        if not self.class_names:
            raise ValueError("Model classes are not initialized. Define weights_path while creating the object.")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        return self.class_names[predicted_idx.item()]