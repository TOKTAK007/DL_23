#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
import class_functions as C
#-----------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#
from torchvision import datasets, transforms
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

# NOTE predifined parameters
batch_size = 128 # depends on GPU memory size
learning_rate = 0.01
momentum = 0.5
epochs = 5

'''
step 1: load mnist dataset and preview
'''

# NOTE training data
train_dataset = datasets.MNIST('../datasets/', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# NOTE validation data
validation_dataset = datasets.MNIST('../datasets/', train=False, transform=transforms.ToTensor())
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
# NOTE cheack training data size and type
# for (X_train, y_train) in train_loader:
#     print('X_train:', X_train.size(), 'type:', X_train.type())
#     print('y_train:', y_train.size(), 'type:', y_train.type())
#     break
# # NOTE preview mnist
# C.view_mnist(X_train, y_train)

'''
step 2: call neural network class function and predefine optimization parameters.
'''

model = C.Net().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
criterion = nn.CrossEntropyLoss()
# print(model)

'''
step 3: training data
'''

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    C.train(model, train_loader, device, optimizer, epoch, criterion, log_interval=100)
    C.validate(model, validation_loader, device, criterion, lossv, accv)

#! FIXME need to store training and validation
plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs+1), lossv)
plt.title('validation loss')

plt.figure(figsize=(5, 3))
plt.plot(np.arange(1, epochs+1), accv)
plt.title('validation accuracy')
plt.show()