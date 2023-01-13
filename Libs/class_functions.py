#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as TF
#-----------------------------------------------------------------------------------------#
from torch import nn
#-----------------------------------------------------------------------------------------#
plt.rcParams.update({'font.size': 22})
#-----------------------------------------------------------------------------------------#

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
	plt.figure(figsize=(25, 16))
	plt.scatter(train_data, train_labels, c='blue', s=70, edgecolors='black', linewidths=3, alpha=1.0, marker='o', label='training')
	plt.scatter(test_data, test_labels, c='red', s=70, edgecolors='black', linewidths=3, alpha=1.0, marker='o', label='validation')
	if predictions is not None:
		plt.scatter(test_data, predictions, c='green', s=70, edgecolors='black', linewidths=3, alpha=1.0, marker='D', label='predicting')
	plt.legend()
	plt.show()

class LinearRegressionModel(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) 
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True) 
    def forward(self, x: torch.Tensor): 
        return self.weights * x + self.bias 

def MSE(data_y, model):
	sum = 0.
	for i in range (0, len(data_y)):
		sum += (data_y[i] - model[i])**2
	return sum/len(data_y)

def loss_curves(epoch_count, train_loss_values, test_loss_values):
	plt.figure(figsize=(15, 9))
	plt.plot(epoch_count, train_loss_values, label='train')
	plt.plot(epoch_count, test_loss_values, label='validation')
	plt.title('Lost Curves (Mean Absolute Error)', fontweight='bold')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt.show()

def view_mnist(X_train, y_train):
	plt.figure(figsize=(25, 5))
	for i in range(10):
		plt.subplot(1, 10, i+1)
		plt.axis('off')
		plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap='gray_r')
		plt.title('Class: ' + str(y_train[i].item()), fontweight='bold')
	plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = TF.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = TF.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return TF.log_softmax(self.fc3(x), dim=1)

def train(model, train_loader, device, optimizer, epoch, criterion, log_interval=200):
    # Set model to training mode
    model.train()
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)
        # Zero gradient buffers
        optimizer.zero_grad() 
        # Pass data through the network
        output = model(data)
        # Calculate loss
        loss = criterion(output, target)
        # Backpropagate
        loss.backward()
        # Update weights
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validate(model, validation_loader, device, criterion, loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)
    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))

#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out