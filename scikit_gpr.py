import pickle
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import deal_with_files
import constants as c
import binning

NTag = 4
xlist, ylist, hlist = deal_with_files.load_1d(NTag)
xmesh, ymesh, hmesh = deal_with_files.load_mesh(NTag)

X = np.array(list(zip(xlist, ylist)))
y = hlist

# Input space
x1 = c.xbins #p
x2 = c.ybins #q

C_var = 1.0
RBF_var = 2
kernel = C(C_var, (1e-3, 1e3)) * RBF([RBF_var,RBF_var], (1e-2, 1e2))

print('creating')
gp = GaussianProcessRegressor(kernel=kernel)#, n_restarts_optimizer=15)
print('fitting')
print(X[0])
in_SR = binning.binInSR(X[:,0], X[:,1])
X = X[np.logical_not(in_SR)]
y = y[np.logical_not(in_SR)]

gp.fit(X, y)
print('producting')
x1 = np.linspace(X[:,0].min(), X[:,0].max()) #p
x2 = np.linspace(X[:,1].min(), X[:,1].max()) #q
print(len(x1), len(x2), xmesh.shape)
x1x2 = np.array(list(product(x1, x2)))
print('predicting')
y_pred, MSE = gp.predict(x1x2, return_std=True)
print('reshaping')
X0p = x1x2[:,0].reshape(len(x1),len(x2))
X1p = x1x2[:,1].reshape(len(x1),len(x2))
Zp = np.reshape(y_pred,(len(x1),len(x2)))
print('plotting')
fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(2*7,7))
axes[1].pcolormesh(X0p, X1p, Zp)
axes[1].set_title(f'GPs, Kernel = C{C_var} * R{RBF_var}')
axes[0].pcolormesh(xmesh, ymesh, hmesh)
axes[0].set_title('Data')
plt.savefig(f'figures{c.bin_sizes}/gp_prediction_{NTag}tag.png')
#plt.show()

with open('Zp.p', 'wb') as zfile:
    pickle.dump(Zp, zfile)