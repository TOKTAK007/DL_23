#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import pytorch_lightning_libs as PLL
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import time
#-----------------------------------------------------------------------------------------#
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchsummary import summary
#-----------------------------------------------------------------------------------------#

'''
Step 0: Predefined Parameters.
'''

DATASET_PATH = '../larger_than_50_MB/team_datasets/Aum/datasets'
train_size = 0.7; val_size = 0.2; test_size = 0.1
seed = 53
num_workers = 24 # CPU
batch_size = 32
model_name = 'ResNet44'
CHECKPOINT_PATH = '../larger_than_50_MB/save_trained_model/' + model_name + '.ckpt'
output_dim = 2
image_dim_1 = 32; image_dim_2 = 32
epochs = 10
image2plot = 25
device = PLL.set_seed(seed)

'''
Step 1: Splitting the Dataset and Viewing Images. 
'''

class_names = os.listdir(DATASET_PATH)
print('class names: ', class_names)
num_class = len(class_names)
image_files=glob.glob(DATASET_PATH + '/*/*.png', recursive=True)
print('total images in: ', DATASET_PATH, ' is ', len(image_files))
idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key, value in idx_to_class.items()}
train_idx, test_idx, val_idx = random_split(image_files, [train_size, val_size, test_size])
train_list=[image_files[i] for i in train_idx.indices]
val_list=[image_files[i] for i in test_idx.indices]
test_list=[image_files[i] for i in val_idx.indices]
print('number of training images: ', len(train_list),
	'\nnumber of val images: ', len(val_list),
	'\nnumber of test images: ', len(test_list))
PLL.view_images(train_list, num_class)

'''
Step 2: Data Preprocessing
'''

mean, std = PLL.means_std(train_list)
flip = transforms.RandomHorizontalFlip()
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean, std)
transform_train = transforms.Compose([flip, to_tensor, normalize])
transform_val = transforms.Compose([to_tensor, normalize])
transform_test = transforms.Compose([to_tensor, normalize])
train_dataset = PLL.SatelliteDataset(train_list, class_to_idx, transform_train)
val_dataset = PLL.SatelliteDataset(val_list, class_to_idx, transform_val)
test_dataset = PLL.SatelliteDataset(test_list, class_to_idx, transform_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
    shuffle=True, drop_last=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
    shuffle=False, drop_last=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
    shuffle=False, drop_last=False, num_workers=num_workers)
PLL.view_images_from_loader(train_loader, num_class, class_names, mean=mean, std=std)

'''
Step 3: Model Initialization and Setup. 

Architecture choices: ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202 
'''

ResNet_config = U.ResNet_achitecture_choices(model_name)
model = NNA.ResNet(ResNet_config, output_dim)
print(model.to(device))
summary(model, (3, image_dim_1, image_dim_2))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

'''
Step 4: Training the Model.
'''

history_train_loss = []
history_valid_loss = []
best_valid_loss = float('inf')
for epoch in range(epochs):
    start_time = time.monotonic()
    train_loss, train_acc = U.train(model, train_loader, optimizer, criterion, device)
    valid_loss, valid_acc = U.evaluate(model, val_loader, criterion, device)
    history_train_loss.append(train_loss)
    history_valid_loss.append(valid_loss)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
    end_time = time.monotonic()
    epoch_mins, epoch_secs = U.epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
U.loss_history_plot(history_train_loss, history_valid_loss, model_name)

'''
Step 6: Plotting Confusion Matrix.
'''

model.load_state_dict(torch.load(CHECKPOINT_PATH))
test_loss, test_acc = U.evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
images, labels, probs = U.get_predictions(model, test_loader, device)
pred_labels = torch.argmax(probs, 1)
U.plot_confusion_matrix_CIFAR10(labels, pred_labels, class_names)

'''
Step 7: Plotting the Most Incorrect Prediction.
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
