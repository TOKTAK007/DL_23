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
