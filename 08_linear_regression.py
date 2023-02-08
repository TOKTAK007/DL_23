import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

'''
step 1: input data
'''

# NOTE input data
weights = np.array([99.79, 117.93, 95.25, 100.70, 108.86, 87.54]) # axis-x
heights = np.array([198.12, 215.90, 205.74, 203.20, 205.74, 190.05]) # axis-y
# 1.9339385169782524 -0.2534447457805413
m = 0.58 # slope
c = 145 # intercept
y = m*weights + c

# sdfsdf
# DEBUG_BYTECODE_SUFFIXES

'''
step 2: compute error
'''

# error = MAE(heights, y)
# print(error)
error = F.MSE(heights, y)
print(error)

'''
step 3: plot
'''

# NOTE plot
plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 18})
plt.scatter(weights, heights, s=300, c='orange', edgecolors='black', alpha=1.0, marker='o')
plt.plot(weights, y, 'r')
plt.title('Mean Square Error: ' + str('%.4f' % error), fontweight='bold')
plt.xlabel('weight', fontweight='bold')
plt.ylabel('height', fontweight='bold')
# plt.savefig('image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()
