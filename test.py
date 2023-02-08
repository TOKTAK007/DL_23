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
#-----------------------------------------------------------------------------------------#
from torchvision import datasets, transforms
from torchsummary import summary
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

# NOTE predifined parameters
download_folder = '../larger_than_50_MB/'
batch_size = 128 # depends on GPU memory size
image2plot = 25
val_ratio = 0.8
image_dim_1 = 32; image_dim_2 = 32
output_dim = 10 # should equal to number of classes
learning_rate = 0.01
momentum = 0.5
epochs = 10
save_model = '../larger_than_50_MB/save_trained_model/AlexNet_custom_dataset.pt'
