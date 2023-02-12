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
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
#-----------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', device)
#-----------------------------------------------------------------------------------------#

'''
Step 0: Setting Up the Environment and Predefing Parameters. 
'''

# NOTE fix random seed to reproducible results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# NOTE predifined parameters
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
epochs = 10
save_model = '../larger_than_50_MB/save_trained_model/test.pt'

'''
step 1: Accessing and Splitting the Image Data into Training, Testing, and Validation Sets.
This step needs to accessing the main folder and manipulate as the following:

1) Accessing the main folder that contains subclasses using the os.listdir() function and storing the subclasses in the class_names variable.
2) Printing the number of subclasses stored in class_names.
3) Finding all the PNG image files in the subclasses using the glob.glob() function and storing the file paths in the image_files variable.
4) Printing the total number of images found in the data_dir folder.
5) Creating two dictionaries, idx_to_class, and class_to_idx, for mapping between the indices of the subclasses and their names.
6) Splitting the image files into training, test, and validation sets using the random_split() function and storing the indices of each set in the variables train_idx, test_idx, and val_idx, respectively.
7) Creating separate lists of image paths for each set using the indices.
8) Printing the number of images in each set.
9) Finally, there is a commented-out call to the imshow_numpy_format() function, which is used to view input images.
'''

# NOTE 1-4
class_names = os.listdir(data_dir)
print('class names: ', class_names)
num_class = len(class_names)
image_files=glob.glob(data_dir + '/*/*.png', recursive=True)
print('total images in: ', data_dir, ' is ', len(image_files))
# NOTE 5-8
idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key, value in idx_to_class.items()}
train_idx, test_idx, val_idx = random_split(image_files, [train_size, test_size, val_size])
train_list=[image_files[i] for i in train_idx.indices]
test_list=[image_files[i] for i in test_idx.indices]
val_list=[image_files[i] for i in val_idx.indices]
print('number of training images: ', len(train_list),
      '\nnumber of test images: ', len(test_list),
      '\nnumber of val images: ', len(val_list))
# NOTE 9
# U.imshow_numpy_format(train_list)

'''
step 2: Data Augmentation and Iterator Creation.

The title of step 2 can be "Data augmentation and iterator creation."
This step involves the creation of PyTorch data iterators for the training, validation, and test datasets. The data augmentation uses the U.augmentation function, which performs data normalization and augmentation (mean subtraction and standard deviation scaling). The function takes the mean RGB values, standard deviation RGB values, the training data list, the mapping from class index to class names, the validation data list, and the test data list as inputs. The function outputs three PyTorch data iterators for the training, validation, and test datasets.
'''

# NOTE Data augmentation
train_iterator, valid_iterator, test_iterator = U.augmentation(meanRGB,
                                                               stdRGB,
                                                               train_list,
                                                               class_to_idx,
                                                               val_list,
                                                               test_list)
# NOTE Batching the training data
dataiter = iter(train_iterator)
inputs, classes = next(dataiter)
class_list = [class_names[x] for x in classes]
# NOTE Making a grid from batch
out = torchvision.utils.make_grid(inputs)
# NOTE Quick view torch format images
# U.quick_show_torch(out, meanRGB, stdRGB, title=class_list)

'''
step 3: Model Initialization and Parameter Selection. 

In this step, the code creates an instance of the AlexNet_64 model, a modified version of the AlexNet model specifically adapted for image classification problems where the input images are 64x64 in size. Next, the code prints the model structure and uses the summary function to display the shape and size of each layer in the model. Finally, the optimizer is initialized using the Adam optimizer, and the loss function is specified using the Cross-Entropy Loss. Both the model and the criterion are then moved to the device specified by the device.
'''

model = NNA.AlexNet_64(output_dim)
print(model.to(device))
summary(model, (3, image_dim_1, image_dim_2))
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
model = model.to(device)
criterion = criterion.to(device)

'''
step 4: Training and Evaluating Model. 

This code performs the training and evaluation loop for the number of specified epochs. The loop contains the following steps:
1) best_valid_loss is initialized to a large value (positive infinity). This value is used later to keep track of the best validation loss seen during training, which will be used to determine if the model has improved.
2) The training loop starts with for epoch in range(epochs), which means that the training process will repeat epochs times.
3) The start time is recorded using start_time = time.monotonic().
4) The training loss and accuracy are computed using the U.train function, which takes as input the model, the training data iterator, the optimizer, the loss criterion, and the device (CPU or GPU). The function returns the average training loss and accuracy over the training data.
5) The validation loss and accuracy are computed using the U.evaluate function, which is similar to the U.train function but is applied to the validation data instead of the training data.
6) If the current validation loss is lower than the best validation loss seen so far, best_valid_loss is updated, and the model's parameters are saved using torch.save. This is because a lower validation loss indicates that the model has improved, and we want to save the best model seen during training.
7) The end time is recorded using end_time = time.monotonic().
8) The elapsed time for the current epoch is calculated using the U.epoch_time function, which takes the start and end times as inputs.
9) Finally, the training progress is printed, including the epoch number, epoch time, training loss, training accuracy, validation loss, and validation accuracy.
10) After the loop has finished, the function U.loss_history_plot is called with history_train_loss and history_valid_loss as inputs, which will plot the loss history of the training and validation sets throughout training.
'''

# NOTE allocate memory for training and validation history plots
history_train_loss = []
history_valid_loss = []
# NOTE 1
best_valid_loss = float('inf')
# NOTE 2
for epoch in range(epochs):
	# NOTE 3-5
    start_time = time.monotonic()
    train_loss, train_acc = U.train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = U.evaluate(model, valid_iterator, criterion, device)
    history_train_loss.append(train_loss)
    history_valid_loss.append(valid_loss)
	# NOTE 6
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), save_model)
	# NOTE 7
    end_time = time.monotonic()
	# NOTE 8
    epoch_mins, epoch_secs = U.epoch_time(start_time, end_time)
	# NOTE 9
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
# NOTE 10
U.loss_history_plot(history_train_loss, history_valid_loss)

'''
step 5: Evaluating Trained Model using Confusion Matrix.

1) The saved model from the previous step is loaded using torch.load(save_model) and loaded into the model.
2) The test set is evaluated using the U.evaluate function, which returns the test loss and test accuracy. These values are printed to the console.
3) The predictions are obtained for the test set using the U.get_predictions function, which returns the images, actual labels, and predicted probabilities.
4) The predicted labels are obtained by taking the argmax of the predicted probabilities along the 1st dimension.
5) The confusion matrix is plotted using the U.plot_confusion_matrix_CIFAR10 function, which takes the actual labels, predicted labels, and class names as inputs. The confusion matrix is used to evaluate the model's performance by showing which classes are often confused with each other.
'''

model.load_state_dict(torch.load(save_model))
test_loss, test_acc = U.evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
images, labels, probs = U.get_predictions(model, test_iterator, device)
pred_labels = torch.argmax(probs, 1)
U.plot_confusion_matrix_CIFAR10(labels, pred_labels, class_names)

'''
step 6: Visualizing the Most Incorrect Predictions Based on Probabilities.

1) The correct predictions are obtained by comparing the actual labels to the predicted labels using the torch.eq function.
2) A list of incorrect examples is created by looping through the images, labels, probabilities, and correct predictions. If the prediction is not correct, the example is added to the list of incorrect examples.
3) The incorrect examples are sorted in descending order based on the maximum predicted probability using the sort method.
4) The U.plot_most_incorrect_CIFAR10 function is called to plot the most incorrect examples. This function takes the list of incorrect examples, class names, and the number of images to plot as inputs, which helps to understand the model's weaknesses and identify potential areas for improvement.
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