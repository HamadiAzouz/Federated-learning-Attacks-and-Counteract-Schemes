# models/simple_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomFashionModel(nn.Module):
    def __init__(self):
        super(CustomFashionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes for FashionMNIST

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def get_model_parameters(self):
        return [param.detach().cpu().numpy() for param in self.parameters()]

    def set_model_parameters(self, parameters):
        for param, new_param in zip(self.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.data.dtype, device=param.data.device)

    def train_epoch(self, data_loader, criterion, optimizer, device, poison_labels=False):
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Data poisoning: label flipping
            if poison_labels:
                labels = 9 - labels  # Flip all labels (for FashionMNIST: 0<->9, 1<->8, ...)
            outputs = self(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def test_epoch(self, data_loader, criterion, device):
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy
