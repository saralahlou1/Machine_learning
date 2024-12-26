import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights


class SimpleCNN(nn.Module):
    num_pixels: int

    def __init__(self, num_pixels, epochs=100):
        super(SimpleCNN, self).__init__()
        self.num_pixels = num_pixels
        self.epochs = epochs
        self.accuracy_per_epoch = []
        self.test_accuracy_per_epoch = []

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * num_pixels * num_pixels // 16, 256)
        self.fc2 = nn.Linear(256, 11)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * self.num_pixels * self.num_pixels // 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PreTrained:
    def __init__(self, layer_sizes, activations, epochs=100):
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.accuracy_per_epoch = []
        self.test_accuracy_per_epoch = []
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.freeze()
        self.set_fc_layers()

    def freeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def set_fc_layers(self):
        fc_layers = []
        input_dim = 512
        fc_layers.append(nn.Flatten())
        for output_dim, activation in zip(self.layer_sizes, self.activations):
            fc_layers.append(nn.Linear(input_dim, output_dim))
            fc_layers.append(activation)
            input_dim = output_dim

        self.resnet.fc = nn.Sequential(*fc_layers)


def train_model(whole_object, model, _criterion, _optimizer, _train_loader, _test_loader,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.to(device)
    for epoch in tqdm(range(whole_object.epochs), total=whole_object.epochs, leave=True, desc="Epochs loop ..."):
        model.train()
        for images, labels in tqdm(_train_loader, total=len(_train_loader), leave=False, desc="Mini batch loop ..."):
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(1).long()
            _optimizer.zero_grad()
            outputs = model(images)
            loss = _criterion(outputs, labels)
            loss.backward()
            _optimizer.step()
        whole_object.accuracy_per_epoch.append(compute_accuracy(model, _train_loader, device=device))
        whole_object.test_accuracy_per_epoch.append(compute_accuracy(model, _test_loader, device=device))


def compute_accuracy(model, data_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.squeeze(1)).sum().item()

    accuracy = correct / total
    return accuracy
