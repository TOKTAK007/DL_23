#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import time
import torchvision.models as TM
#-----------------------------------------------------------------------------------------#
from torchvision import datasets, transforms
from torchsummary import summary
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

# NOTE predifined parameters
download_folder = '../datasets/'
batch_size = 128 # depends on GPU memory size
image2plot = 25
val_ratio = 0.8
image_dim_1 = 32; image_dim_2 = 32
output_dim = 10 # should equal to number of classes
learning_rate = 0.01
momentum = 0.5
epochs = 300
save_model = '../save_trained_model/VGG.pt'

'''
step 1: load CIFAR10.
'''

train_data = datasets.CIFAR10(root=download_folder,
                              train=True,
                              download=True)

'''
step 2: normalize data.
'''

# NOTE compute mean and standard diviation.
means = train_data.data.mean(axis=(0, 1, 2)) / 255
stds = train_data.data.std(axis=(0, 1, 2)) / 255
print(f'Calculated means: {means}')
print(f'Calculated stds: {stds}')
# NOTE transform to PyTorch format
train_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=means, std=stds)])
test_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=means, std=stds)])
train_data = datasets.CIFAR10(download_folder,
                              train=True,
                              download=True,
                              transform=train_transforms)
test_data = datasets.CIFAR10(download_folder,
                             train=False,
                             download=True,
                             transform=test_transforms)
# NOTE store class names for plotting
classes = train_data.classes
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# # NOTE preview some images
# images, labels = zip(*[(image, label) for image, label in
#                        [train_data[i] for i in range(image2plot)]])
# # NOTE image without appling normalization
# U.plot_CIFAR10(images, labels, classes)
# # NOTE image with appling normalization
# U.plot_CIFAR10(images, labels, classes, normalize=True)

'''
step 3: divide some parts of the training data for validation. Remember that validation data should be separated from training data. This data set will be used for optimization purposes, whereas test data (unseen data) will be used for evaluating model performance.
'''

# NOTE devide some parts of the training data for validation
n_train_examples = int(len(train_data) * val_ratio)
n_valid_examples = len(train_data) - n_train_examples
# NOTE random split
train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')
# # NOTE preview some images
# images, labels = zip(*[(image, label) for image, label in
#                        [valid_data[i] for i in range(image2plot)]])
# U.plot_CIFAR10(images, labels, classes, normalize=True)

'''
step 4: split training, validation, and test data to fit GPU/CPU. This technique helps parallel computation. The size of the splitting depends on the cores in GPU/CPU.
'''

train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_iterator = data.DataLoader(valid_data, batch_size=batch_size)
test_iterator = data.DataLoader(test_data, batch_size=batch_size)

'''
step 5: configure VGG architectures.
'''

vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M',
                512, 512, 'M']
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
                'M', 512, 512, 512, 'M']
vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
# NOTE select one VGG configuration to train the model
vgg_layers = U.get_vgg_layers(vgg19_config, batch_norm=True)
model = NNA.VGG(vgg_layers, output_dim)

'''
step 6: download trained weights from PyTorch. Once the weight model is downloaded, this means all layers are frozen. The default of the training weights is from IMAGENET, having 1000 classes, so we need to modify the last layer to fit with our number of classes in the dataset. The CIFAR10 contains 10 classes; hence the last layer will reduce from 1000 to 10.
'''

weights = TM.VGG19_BN_Weights.DEFAULT # .DEFAULT = best available weights 
pretrained_model = TM.vgg19_bn(weights=weights)
print(pretrained_model.classifier[-1])
IN_FEATURES = pretrained_model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, output_dim)
pretrained_model.classifier[-1] = final_fc
print(pretrained_model.classifier)
model.load_state_dict(pretrained_model.state_dict())
for parameter in model.features.parameters():
    parameter.requires_grad = False
for parameter in model.classifier[:-1].parameters():
    parameter.requires_grad = False

'''
step 7: preview the model architecture.
'''

print(pretrained_model.to(device))
summary(pretrained_model, (3, image_dim_1, image_dim_2))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

'''
step 8: train and validate neural network architecture. Remember, we need to create a folder to store the trained model.
'''

# NOTE predfine history plot
history_train_loss = []
history_valid_loss = []
# NOTE main loop for training
best_valid_loss = float('inf')
for epoch in range(epochs):
    start_time = time.monotonic()
    train_loss, train_acc = U.train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = U.evaluate(model, valid_iterator, criterion, device)
    history_train_loss.append(train_loss)
    history_valid_loss.append(valid_loss)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), save_model)
    end_time = time.monotonic()
    epoch_mins, epoch_secs = U.epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
# NOTE plot history
U.loss_history_plot(history_train_loss, history_valid_loss)

'''
step 9: load the trained model and evaluate it, using confusion matrix.
'''

model.load_state_dict(torch.load(save_model))
test_loss, test_acc = U.evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
images, labels, probs = U.get_predictions(model, test_iterator, device)
pred_labels = torch.argmax(probs, 1)
U.plot_confusion_matrix(labels, pred_labels)

'''
step 10: display the most incorrect predictions based on probabilities.
'''

corrects = torch.eq(labels, pred_labels)
incorrect_examples = []
for image, label, prob, correct in zip(images, labels, probs, corrects):
    if not correct:
        incorrect_examples.append((image, label, prob))
incorrect_examples.sort(reverse = True, key=lambda x: torch.max(x[2], dim=0).values)
U.plot_most_incorrect_CIFAR10(incorrect_examples, classes, image2plot)