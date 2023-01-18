#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.optim as optim
import time
import torch.nn.functional as TF
#-----------------------------------------------------------------------------------------#
from torchvision import datasets, transforms
from torchsummary import summary
# from tqdm.notebook import trange, tqdm
from tqdm import tqdm
from sklearn import metrics
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

# NOTE predifined parameters
ROOT = '../datasets/'
batch_size = 128 # depends on GPU memory size
learning_rate = 0.01
momentum = 0.5
epochs = 5

'''
step 1: load mnist dataset and preview
'''

# # NOTE training data
# train_transforms = transforms.Compose([transforms.ToTensor()])
# train_data = datasets.MNIST(root=ROOT,
#                             train=True,
#                             download=True,
#                             transform=train_transforms)
# NOTE test data
test_transforms = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root=ROOT,
                            train=True,
                            download=True,
                            transform=test_transforms)

# # print(f'Number of training examples: {len(train_data)}')
# # print(f'Number of testing examples: {len(test_data)}')

# # N_IMAGES = 25
# # images = [image for image, label in [train_data[i] for i in range(N_IMAGES)]]
# # F.plot_images(images)

# val_ratio = 0.8
# n_train_examples = int(len(train_data) * val_ratio)
# n_valid_examples = len(train_data) - n_train_examples

# train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
# # print(f'Number of training examples: {len(train_data)}')
# # print(f'Number of validation examples: {len(valid_data)}')
# # print(f'Number of testing examples: {len(test_data)}')

# # N_IMAGES = 25
# # images = [image for image, label in [valid_data[i] for i in range(N_IMAGES)]]
# # F.plot_imaes(images)

BATCH_SIZE = 256
# train_iterator = data.DataLoader(train_data,
#                                  shuffle=True,
#                                  batch_size=BATCH_SIZE)
# valid_iterator = data.DataLoader(valid_data,
#                                  batch_size=BATCH_SIZE)
test_iterator = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)

INPUT_DIM = 28 * 28
OUTPUT_DIM = 10
model = NNA.MLP(INPUT_DIM, OUTPUT_DIM)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f'The model has {count_parameters(model):,} trainable parameters')
# # print(model)
# # summary(model, (1, 28, 28))
# optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
# def train(model, iterator, optimizer, criterion, device):
	
#     epoch_loss = 0
#     epoch_acc = 0

#     model.train()

#     for (x, y) in tqdm(iterator, desc="Training", leave=False):

#         x = x.to(device)
#         y = y.to(device)

#         optimizer.zero_grad()

#         y_pred, _ = model(x)

#         loss = criterion(y_pred, y)

#         acc = calculate_accuracy(y_pred, y)

#         loss.backward()

#         optimizer.step()

#         epoch_loss += loss.item()
#         epoch_acc += acc.item()

#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
	
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs

# EPOCHS = 10

# best_valid_loss = float('inf')
# for epoch in range(EPOCHS):
#     start_time = time.monotonic()
#     train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
#     valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'tut1-model.pt')
#     end_time = time.monotonic()
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#     print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('tut1-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

def get_predictions(model, iterator, device):
	
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = TF.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs

images, labels, probs = get_predictions(model, test_iterator, device)

pred_labels = torch.argmax(probs, 1)

def plot_confusion_matrix(labels, pred_labels):
	
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(10))
    cm.plot(values_format='d', cmap='Greens', ax=ax)
    plt.show()

plot_confusion_matrix(labels, pred_labels)


corrects = torch.eq(labels, pred_labels)
incorrect_examples = []

for image, label, prob, correct in zip(images, labels, probs, corrects):
    if not correct:
        incorrect_examples.append((image, label, prob))

incorrect_examples.sort(reverse=True,
                        key=lambda x: torch.max(x[2], dim=0).values)

def plot_most_incorrect(incorrect, n_images):
	
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 10))
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        image, true_label, probs = incorrect[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        ax.imshow(image.view(28, 28).cpu().numpy(), cmap='bone')
        ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n'
                     f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    plt.show()

N_IMAGES = 25

plot_most_incorrect(incorrect_examples, N_IMAGES)
