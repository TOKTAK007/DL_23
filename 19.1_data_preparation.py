#-----------------------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np
import os
#-----------------------------------------------------------------------------------------#
from PIL import Image
#-----------------------------------------------------------------------------------------#

PATH = '../larger_than_50_MB/input_data/'
DATASETS = '../larger_than_50_MB/datasets'
class_1 = 'land'
class_2 = 'water'
image_dim_1 = 32; image_dim_2 = 32
threshold = 0.9*(image_dim_1*image_dim_2)

if os.path.exists(DATASETS):
	print(f'already created {DATASETS}')
elif not os.path.exists(DATASETS):
	os.mkdir(DATASETS)
	print(f'created {DATASETS} folder')
	if not os.path.exists(DATASETS + class_1):
		os.mkdir(DATASETS + '/' + class_1)
		print(f'created {class_1} folder')
	if not os.path.exists(DATASETS + class_2):
		os.mkdir(DATASETS + '/' + class_2)
		print(f'created {class_2} folder')

image = Image.open(PATH + 'image.png')
image = np.array(image) # RGBA
image = image[:, :, :3]
label = Image.open(PATH + 'label.png')
label = np.array(label)
label = label[:, :, 0]
label = np.where(label > 0, 1, 0)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
# ax1.imshow(image)
# ax1.set_title('image Image')
# ax2.imshow(label)
# ax2.set_title('Label Image')
# plt.show()

axisY, axisX, _ = image.shape
image_size = image_dim_1 * image_dim_2
num_crop_x = axisX // image_dim_1
num_crop_y = axisY // image_dim_2

count = 1
for i in range(num_crop_y):
	for ii in range(num_crop_x):
		begX = ii * image_dim_2
		begY = i * image_dim_1
		endX = min(begX + image_dim_2, axisX)
		endY = min(begY + image_dim_1, axisY)
		coordinate = f'{begY}_{begX}' # referent point is top left
		crop_image = image[begY:endY, begX:endX, :]
		crop_label = label[begY:endY, begX:endX]
		if image_size == crop_image.shape[0]*crop_image.shape[1]:
			vector_size = crop_label.flatten()
			class_type = []
			if len(vector_size[vector_size != 0]) > threshold:
				class_type = class_1
			else:
				class_type = class_2
			if class_type:
				crop_image = Image.fromarray(crop_image)
				crop_image.save(f'{DATASETS}/{class_type}/img_{count:06}_{coordinate}.png')
				print(f'{DATASETS}/{class_type}/img_{count:06}_{coordinate}.png')
				count += 1
		else:
			print('incorrect image size')		