#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import torchvision
#-----------------------------------------------------------------------------------------#
from torch.utils.data import random_split
from torchsummary import summary
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

'''
Step 0: Setting Up the Environment and Predefing Parameters. 
'''

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

data_dir = 'LFS_folders/sat_image_classification' # where is the main folder containing subclasses
train_size = 0.7; test_size = 0.1; val_size = 0.2
meanRGB = [0.485, 0.456, 0.406]
stdRGB = [0.229, 0.224, 0.225]
batch_size = 128 # depends on GPU memory size
image2plot = 25
image_dim_1 = 64; image_dim_2 = 64
output_dim = 2 # should equal to number of classes
learning_rate = 0.01
momentum = 0.5
ResNet_achitecture = 'ResNet44'
epochs = 10
save_model = '../larger_than_50_MB/save_trained_model/' + ResNet_achitecture + '.pt'

'''
step 1: Accessing and Splitting the Image Data into Training, Testing, and Validation Sets.
'''

class_names = os.listdir(data_dir)
print('class names: ', class_names)
num_class = len(class_names)
image_files=glob.glob(data_dir + '/*/*.png', recursive=True)
print('total images in: ', data_dir, ' is ', len(image_files))
idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key, value in idx_to_class.items()}
train_idx, test_idx, val_idx = random_split(image_files, [train_size, test_size, val_size])
train_list=[image_files[i] for i in train_idx.indices]
test_list=[image_files[i] for i in test_idx.indices]
val_list=[image_files[i] for i in val_idx.indices]
print('number of training images: ', len(train_list),
      '\nnumber of test images: ', len(test_list),
      '\nnumber of val images: ', len(val_list))
# U.imshow_numpy_format(train_list)

'''
step 2: Data Augmentation and Iterator Creation.
'''

train_iterator, valid_iterator, test_iterator = U.augmentation(meanRGB,
                                                               stdRGB,
                                                               train_list,
                                                               class_to_idx,
                                                               val_list,
                                                               test_list)
dataiter = iter(train_iterator)
inputs, classes = next(dataiter)
class_list = [class_names[x] for x in classes]
out = torchvision.utils.make_grid(inputs)
# U.quick_show_torch(out, meanRGB, stdRGB, title=class_list)

'''
step 3: Model Initialization and Parameter Selection.

Architecture choices: ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202 
'''

ResNet_config = U.ResNet_achitecture_choices(ResNet_achitecture)
model = NNA.ResNet(ResNet_config, output_dim)
print(model.to(device))
summary(model, (3, image_dim_1, image_dim_2))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

'''
step 4: Training and Evaluating Model. 
'''

history_train_loss = []
history_valid_loss = []
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
U.loss_history_plot(history_train_loss, history_valid_loss)

'''
step 5: Evaluating Trained Model using Confusion Matrix.
'''

model.load_state_dict(torch.load(save_model))
test_loss, test_acc = U.evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
images, labels, probs = U.get_predictions(model, test_iterator, device)
pred_labels = torch.argmax(probs, 1)
# U.plot_confusion_matrix_CIFAR10(labels, pred_labels, class_names)

'''
step 6: Visualizing the Most Incorrect Predictions Based on Probabilities.
'''

corrects = torch.eq(labels, pred_labels)
incorrect_examples = []
for image, label, prob, correct in zip(images, labels, probs, corrects):
    if not correct:
        incorrect_examples.append((image, label, prob))
incorrect_examples.sort(reverse = True, key=lambda x: torch.max(x[2], dim=0).values)
if len(incorrect_examples) >= image2plot:
    U.plot_most_incorrect_CIFAR10(incorrect_examples, class_names, image2plot)
else:
    print('reduce the number of image2plot')