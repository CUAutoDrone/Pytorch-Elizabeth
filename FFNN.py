import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt


# Model
class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = nn.Linear(784, 392)
        self.linear_layer2 = nn.Linear(392, 196)
        self.linear_layer3 = nn.Linear(196, 98)
        self.linear_layer4 = nn.Linear(98, 49)
        self.linear_layer5 = nn.Linear(49, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))
        x = F.relu(self.linear_layer3(x))
        x = F.relu(self.linear_layer4(x))
        x = F.relu(self.linear_layer5(x))
        return x


model = FFNN()

# Dataset
train_dataset = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)

test_dataset = torchvision.datasets.MNIST(
    "files/",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)
test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=True)


# Loss Function
cross_entropy_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 5


def train_epoch(model, optimizer, dataloader):
    model.train()  # You have to tell your model to go into "train" mode
    losses = []
    for input, labels in dataloader:
        optimizer.zero_grad()
        output = model(input)
        loss = cross_entropy_loss(output, torch.squeeze(labels).float())
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        accuracy = (output.round() == labels).float().mean()
    print("End of epoch accuracy", float(accuracy))
    return losses  # Typically you want to keep a list of all the batch losses


def test_epoch(model, dataloader):
    model.eval()
    losses = []

    with torch.no_grad():
        for input, labels in dataloader:
            output = model(input)
            loss = cross_entropy_loss(output, torch.squeeze(labels).float())
            losses.append(loss.item())
            accuracy = (output.round() == labels).float().mean()
    print("End of validation epoch accuracy", float(accuracy))
    return losses


training_epoch_losses = []
validation_epoch_losses = []

for epoch in range(num_epochs):
    training_epoch_losses.append(train_epoch(model, optimizer, train_dataloader))
    validation_epoch_losses.append(test_epoch(model, test_dataloader))


plt.plot(range(num_epochs), [np.mean(x) for x in training_epoch_losses])
plt.plot(range(num_epochs), [np.mean(x) for x in validation_epoch_losses])
plt.show()