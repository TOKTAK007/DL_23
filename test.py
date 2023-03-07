#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import pytorch_lightning_libs as PLL
#-----------------------------------------------------------------------------------------#
import os
import glob
import torch
# import pytorch_lightning as pl
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import DataLoader
#-----------------------------------------------------------------------------------------#

DATASET_PATH = '../larger_than_50_MB/datasets'
train_size = 0.7; val_size = 0.2; test_size = 0.1
seed = 53
num_workers = 24 # CPU
batch_size = 32
model_name = 'GoogleNet'
CHECKPOINT_PATH = '../larger_than_50_MB/save_trained_model/' + model_name + '.ckpt'

device = PLL.set_seed(seed)

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
# PLL.view_images(train_list, num_class)
mean, std = PLL.means_std(train_list)

resize = transforms.Resize(256)
crop_train = transforms.RandomCrop(224)
flip = transforms.RandomHorizontalFlip()
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean, std)

transform_train = transforms.Compose([resize, crop_train, flip, to_tensor, normalize])
transform_val = transforms.Compose([to_tensor, normalize])
transform_test = transforms.Compose([to_tensor, normalize])

train_dataset = PLL.SatelliteDataset(train_list, class_to_idx, transform_train)
val_dataset = PLL.SatelliteDataset(val_list, class_to_idx, transform_val)
test_dataset = PLL.SatelliteDataset(test_list, class_to_idx, transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
    shuffle=True, drop_last=True, pin_memory=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
    shuffle=False, drop_last=False, pin_memory=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
    shuffle=False, drop_last=False, pin_memory=True, num_workers=num_workers)

# PLL.view_images_from_loader(train_loader, num_class, class_names, mean=mean, std=std)

import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

class MLP(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

model = MLP(input_size=3*32*32, hidden_size=256, num_classes=2)
gpus = [0, 1]
trainer = pl.Trainer(gpus=gpus, max_epochs=10)
trainer.fit(model, train_loader, val_loader)
trainer.test(test_dataloaders=test_loader)