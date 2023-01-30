#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
import functions as C
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
#-----------------------------------------------------------------------------------------#

'''
step 1: import tiff file.
'''

beforeflood = skio.imread('../datasets/satellite_imagery/input_datasets/sen1_permanence.tif', plugin='tifffile')
beforeflood[np.isnan(beforeflood)] = 0
# F.composite_bands(beforeflood, 0.9) 
# plt.imshow(beforeflood, cmap='gray')
# plt.imsave(fname='../datasets/satellite_imagery/permanentwater.png', arr=beforeflood, cmap='gray', format='png')
# plt.show()

'''
step 2: label data. We classify the data into nonflood land, flood, and permanent water categories. The preparing data is to select an area patch-by-patch, which requires human skills to interpret a data class.
'''

# # zone1 = beforeflood[0:7000, 0:7000, :]
# # crop = zone1[0:64, 0:64, :]
# begX = 50; begY = 200; widthX = 64; hightY = 64

# class_1 = '../datasets/satellite_imagery/flood_classes/nonflood/'
# # class_2 = '../datasets/satellite_imagery/flood_classes/flood/'
# # class_3 = '../datasets/satellite_imagery/flood_classes/permanentwater/'
# save_image = class_1 + '000001' + '.png'
# print('save patch image to: %s' % save_image)

# F.draw_box(beforeflood, 0.9, begX, begY, widthX, hightY, save_image)
# # F.composite_bands(beforeflood, 0.9)