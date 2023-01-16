import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
#-----------------------------------------------------------------------------------------#

'''
step 1: input data
'''

weights = np.array([99.79, 117.93, 95.25, 100.70, 108.86, 87.54]) # axis-x
heights = np.array([198.12, 215.90, 205.74, 203.20, 205.74, 190.05]) # axis-y
# print(weights.shape)
# NOTE plot
# plt.figure(figsize=(10, 8))
# plt.rcParams.update({'font.size': 14})
# plt.scatter(weights, heights, s=300, c='orange', edgecolors='black', alpha=1.0, marker='o')
# plt.title('Weights vs Heights of NBA Players', fontweight='bold')
# plt.xlabel('weight', fontweight='bold')
# plt.ylabel('height', fontweight='bold')
# plt.show()

'''
step 2: compute gradient descent
'''

weights = weights.reshape(len(weights), 1)
heights = heights.reshape(len(heights), 1)

lr = pow(5.85, -9)
n_iter = 4000
theta = np.random.randn(2, 1)
X_b = np.c_[np.ones((len(weights), 1)), weights]
theta, cost_history, theta_history = F.gradient_descent(X_b, heights, theta, lr, n_iter)
print('Theta0:          {:0.3f},\nTheta1:          {:0.3f}'.format(theta[0][0], theta[1][0]))
print('Final cost/MSE:  {:0.3f}'.format(cost_history[-1]))

'''
step 3: plot cost history over iterations
'''

fig,ax = plt.subplots(figsize=(10, 8))
ax.set_ylabel(r'J($\Theta$)')
ax.set_xlabel('Iterations')
_ = ax.plot(range(n_iter), cost_history, 'b.')
plt.title('Cost History Over Iterations', fontweight='bold')
# plt.savefig('../drawing/image_out/' + 'cost_his' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()

'''
step 4: plot the predicted model using gradient descent
'''

pred = X_b.dot(theta)
weights = weights[:, 0]
print(weights)
pred = pred[:, 0]
slope, intercept, _, _, _ = linregress(weights, pred)
error = F.MSE(heights, pred)
# NOTE plot
plt.figure(figsize=(10, 5))
plt.rcParams.update({'font.size': 12})
plt.scatter(weights, heights, s=300, c='orange', edgecolors='black', alpha=1.0, marker='o')
plt.plot(weights, pred, 'r-')
plt.title(' MSE: ' + str('%.4f' % error) +
          ' Slope: ' + str('%.4f' % slope) +
          ' Intercept: ' + str('%.4f' % intercept),
          fontweight='bold')
plt.xlabel('weight', fontweight='bold')
plt.ylabel('height', fontweight='bold')
# plt.savefig('../drawing/image_out/' + 'resultGD' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()