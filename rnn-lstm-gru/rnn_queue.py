import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pickle as pkl
import os
from fastai.vision.all import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# input_size = 784 # 28x28
num_classes = 15
num_epochs = 2
batch_size = 100
learning_rate = 0.01

input_size = 2
sequence_length = 2000
hidden_size = 128
num_layers = 2


path = r'C:\Users\user\workspace\pytorch_code_RNN\pytorch-examples\rnn-lstm-gru'
input_rnn_torch = pkl.load(open(os.path.join(path, 'input_torch.pkl'), 'rb'))
output_rnn_torch = pkl.load(open(os.path.join(path, 'output_torch.pkl'), 'rb'))


dset = list(zip(input_rnn_torch, output_rnn_torch))
dl = DataLoader(dset, batch_size=128)



# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)





# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)

        # or:
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # x: (n, 28, 28), h0: (2, n, 128)

        # Forward propagate RNN
        # out, _ = self.rnn(x.float(), h0.float())
        # or:
        out, _ = self.lstm(x.float(), (h0.float(), c0.float()))

        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)

        out = self.fc(out)
        # out: (n, 10)
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
def log_cross(outputs, labels):

    priority_labels = labels[:, :5]
    capacity_labels = labels[:, 5:10]
    service_labels = labels[:, 10:]

    priority_outputs = outputs[:, :5]
    capacity_outputs = outputs[:, 5:10]
    service_outputs = outputs[:, 10:]
    #
    # m = nn.Softmax(dim=1)
    # priority_outputs = m(priority_outputs)
    # capacity_outputs = m(capacity_outputs)

    priority_loss = criterion(priority_outputs, priority_labels)
    capacity_loss = criterion(capacity_outputs, capacity_labels)

    mae_loss = nn.L1Loss()
    L1_Loss = mae_loss(service_outputs, service_labels)


    return L1_Loss + capacity_loss + priority_loss


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(dl)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dl):
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = log_cross(outputs, labels)  # criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in dl:
#         images = images.reshape(-1, sequence_length, input_size).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')