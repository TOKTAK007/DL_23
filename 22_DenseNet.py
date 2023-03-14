#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import utilities as U
import pytorch_lightning_libs as PLL
import neural_network_architectures as NNA
#-----------------------------------------------------------------------------------------#
import os
import glob
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
#-----------------------------------------------------------------------------------------#
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from types import SimpleNamespace
from torchvision import transforms
#-----------------------------------------------------------------------------------------#
torch.set_float32_matmul_precision('medium')

'''
Step 0: Predefined Parameters.
'''

DATASET_PATH = '../larger_than_50_MB/datasets'
train_size = 0.7; val_size = 0.2; test_size = 0.1
seed = 101
num_workers = 24 # CPU
batch_size = 256
model_name = 'GoogleNet'
CHECKPOINT_PATH = '../larger_than_50_MB/save_trained_model/'
output_dim = 2
image_dim_1 = 32; image_dim_2 = 32
epochs = 5
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
# PLL.view_images(train_list, num_class)

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
# PLL.view_images_from_loader(train_loader, num_class, class_names, mean=mean, std=std)


# model_dict["DenseNet"] = DenseNet

densenet_model, densenet_results = PLL.train_model(model_name="DenseNet",
                                                   CHECKPOINT_PATH=CHECKPOINT_PATH,
                                                   train_loader=train_loader,
                                                   val_loader=val_loader,
                                                   test_loader=test_loader,
                                               model_hparams={"num_classes": 2,
                                                              "num_layers": [6,6,6,6],
                                                              "bn_size": 2,
                                                              "growth_rate": 16,
                                                              "act_fn_name": "relu"},
                                               optimizer_name="Adam",
                                               optimizer_hparams={"lr": 1e-3,
                                                                  "weight_decay": 1e-4})
print("densenet Results", densenet_results)

# tensorboard --logdir /home/pongthep_/main/Chula/2023/DL_23/larger_than_50_MB/save_trained_model/GoogleNet/lightning_logs
