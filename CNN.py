import torch
import torch.nn as nn
# Imports
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Implement CNN architecture


class CNN (nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        #  n_out = ((n_in + 2P - K)/s) + 1 = 28 x 28
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        #  n_out = n_out / 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        # 16 in for Linear due to double pooling in feedFor -MaxPooling reduces the size by half
        #  n_out = ((n_in + 2P - K)/s) + 1 = 28 x 28

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x


# Check model shape
model = CNN()
x = torch.randn(64, 1, 28, 28)
print(model(x).shape)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True,
                               transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False,
                              transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

# Initialize NN
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    print(f'Training -epoch: {epoch}')
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        scores = model(data)
        loss = criterion(scores, targets)
        print(f'score: {loss}')
        print(f'loss: {loss}')

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradDescent (Adam step)
        optimizer.step()

# Check model accuracy on trainting and test set


def check_accuracy(loader, model):

    if loader.dataset.train:
        print(f'Checking accuracy on train set')
    else:
        print(f'Checking accuracy on test set')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            # forward
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples) *100}')
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
