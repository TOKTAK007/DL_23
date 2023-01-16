import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#-----------------------------------------------------------------------------------------#

'''
step 1: input data
'''

weights = np.array([99.79, 117.93, 95.25, 100.70, 108.86, 87.54]) # axis-x
heights = np.array([198.12, 215.90, 205.74, 203.20, 205.74, 190.05]) # axis-y

'''
step 2: predefine variables
'''

size = 400
cost_function = np.zeros(shape=(size, size), dtype=np.float64)
beg_m = 0.5; end_m = 0.6
beg_c = 145; end_c = 148
m = np.linspace(beg_m, end_m, size) # axis x (slope)
c = np.linspace(beg_c, end_c, size) # axis y (intercept)

'''
step 3: loop computing cost function
'''

for i in range (0, len(m)):
	for ii in range (0, len(c)):
		y = m[i]*weights + c[ii]
		error = F.MSE(heights, y)
		cost_function[i, ii] = error

'''
step 4: plot cost function map
'''

plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 16})
plt.imshow(cost_function, cmap='jet', aspect='auto', extent=[beg_m, end_m, beg_c, end_c])
plt.title('Mean Square Error: ' + str('%.4f' % np.amin(cost_function)), fontweight='bold')
plt.xlabel('slope', fontweight='bold')
plt.ylabel('intercept', fontweight='bold')
plt.colorbar()
# plt.savefig('../drawing/image_out/' + 'map' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()

# NOTE optional plot
# fig = go.Figure(data=go.Contour(z=cost_function, colorscale='Electric'))
# fig.write_image('pictures/smooth_color.svg', format='svg')
# fig.show()

'''
step 5: plot the best cost function
'''

m = 0.59 # slope
c = 146.7 # intercept
y = m*weights + c
plt.figure(figsize=(15, 10))
plt.rcParams.update({'font.size': 18})
plt.scatter(weights, heights, s=300, c='orange', edgecolors='black', alpha=1.0, marker='o')
plt.plot(weights, y, 'r')
plt.title('Mean Square Error: ' + str('%.4f' % error), fontweight='bold')
plt.xlabel('weight', fontweight='bold')
plt.ylabel('height', fontweight='bold')
# plt.savefig('../drawing/image_out/' + 'ex_plot' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()