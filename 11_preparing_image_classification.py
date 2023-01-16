#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
import class_functions as C
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
#-----------------------------------------------------------------------------------------#

'''
step 1: import tiff file.
'''

beforeflood = skio.imread('../datasets/satellite_imagery/beforeflood_crop_zone.tif', plugin='tifffile')
# print(beforeflood.shape)
# F.composite_bands(beforeflood[0:7000, 0:7000, :], 0.9) # over memory capacity

'''
step 2: label data. We classify the data into nonflood land, flood, and permanent water categories. The preparing data is to select an area patch-by-patch, which requires human skills to interpret a data class.
'''

zone1 = beforeflood[0:7000, 0:7000, :]
crop = zone1[0:64, 0:64, :]
begX = 50; begY = 200; widthX = 64; hightY = 64

class_1 = '../datasets/satellite_data_preparation/nonflood_land/'
class_2 = '../datasets/satellite_data_preparation/flood/'
class_3 = '../datasets/satellite_data_preparation/permanent_water/'
save_image = class_1 + '000001' + '.png'
print('save patch image to: %s' % save_image)

F.draw_box(zone1, 0.9, begX, begY, widthX, hightY, save_image)
F.composite_bands(crop, 0.9)