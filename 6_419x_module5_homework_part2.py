# -*- coding: utf-8 -*-
"""6.419x_Module5_Homework_Part2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15pvB2zZbxdBJQnctKZWXKRYVxhuCKECH

#Part II - Estimating Flows with Gaussian Processes

#Problem 4.a
"""

from google.colab import drive
drive.mount('/content/drive')

from numpy import genfromtxt
import numpy as np
import pandas as pd
from collections import defaultdict
df = defaultdict(dict)
#my_data = genfromtxt('my_file.csv', delimiter=',')

for i in range(1, 101):
  for direction in ["u", "v"]:
    filename = '/content/drive/MyDrive/Colab Notebooks/Module5_OceanFlow/' + str(i) + direction + ".csv"
    df[i][direction] = pd.read_csv(filename, header = None)
  df[i]["mag"] = np.sqrt((df[i]["u"])**2 + (df[i]["v"])**2)

# rewrite data as multidimensional numpy array
data = np.zeros(100*504*555).reshape(100, 504, 555)
data_u = np.zeros(100*504*555).reshape(100, 504, 555)
data_v = np.zeros(100*504*555).reshape(100, 504, 555)
for i in range(1, 101):
  data[i-1, :, :] = np.array(df[i]["mag"])
  data_u[i-1, :, :] = np.array(df[i]["u"])
  data_v[i-1, :, :] = np.array(df[i]["v"])

data = np.swapaxes(data, 1,2)
data_u = np.swapaxes(data_u, 1,2)
data_v = np.swapaxes(data_v, 1,2)

import random
random.seed(0)

# Forming set of all grid points
grids = []
step = 1
for i in range(0, data_u.shape[1], step):
  for j in range(0, data_u.shape[2], step):
    grids.append((i, j))

corr_coeff_u = defaultdict()
corr_coeff_v = defaultdict()

# Randomly select the points
for _ in range(100000):
  x1, y1 = random.choice(grids)
  x2, y2 = random.choice(grids)
  corr_coeff_u[(x1, y1, x2, y2)] = np.corrcoef(data_u[:, x1, y1], data_u[:, x2, y2], rowvar=False)
  corr_coeff_v[(x1, y1, x2, y2)] = np.corrcoef(data_v[:, x1, y1], data_v[:, x2, y2], rowvar=False)

from sklearn.datasets import make_classification
# define dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
print(X.shape)
print(y.shape)

x0, y0 = (1500, 600)
#x0, y0 = (102, 357)
flow_u = data_u[:, int(x0/3), int(y0/3)]
flow_v = data_v[:, int(x0/3), int(y0/3)]
y = flow_u
X =np.arange(100).reshape(100,1)

import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(1, 1)
ax1.plot(range(100), flow_u, color = 'r')
ax1.set_title("horizontal direction coefficient")
plt.show()

fig, ax1 = plt.subplots(1, 1)
ax1.plot(range(100), flow_v, color = 'r')
ax1.set_title("vertical direction coefficient")
plt.show()

def Kernel_periodic(x1, x2, var, l, p):
  return var*np.exp(-2*(np.sin(np.pi*abs(x1-x2)/p)**2)/l**2)

X - X.T

# Calculating SE Kernel
np.identity(2)

sigma11 - sigma12 @ np.linalg.inv(sigma22 + tau*np.identity(len(train_index))) @ sigma21

from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
kf = KFold(n_splits=10, shuffle = True)
tau = 0.001
score_max = 0
for var in [1, 4, 8, 16, 32, 64, 128]:
  for time_index in range(1, 51):
    time = time_index / 10
    l = time * 72
    sigma = var*np.exp(-(X - X.T)**2/l**2)
    #CHOOSE A DIFFERENT KERNEL
    #kernel = var*np.exp(-0.5*np.linalg.norm())
    score = 0
    for train_index, test_index in kf.split(X):
      #print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]
      
      sigma11 = sigma[np.ix_(test_index, test_index)]
      sigma12 = sigma[np.ix_(test_index, train_index)]
      sigma21 = sigma[np.ix_(train_index, test_index)]
      sigma22 = sigma[np.ix_(train_index, train_index)]
      mu1 = np.mean(y_test)
      mu2 = np.mean(y_train)

      mu1_2 = mu1 + sigma12 @ np.linalg.inv(sigma22 + tau*np.identity(len(train_index))) @ (X_train - mu2)
      mu1_2 = mu1_2.reshape(10,1)
      #X_test = X_test.reshape(10, 1)
      #print(mu1_2.shape)
      #print(X_test.shape)
      #print(y_test.shape)
      #print(sigma11_2.shape)

      sigma11_2 = sigma11 - sigma12 @ np.linalg.inv(sigma22 + tau*np.identity(len(train_index))) @ sigma21
      #rv = (mu1_2, sigma11_2)
      #print(rv.pdf(X_test))
      #break
      #score += multivariate_normal.pdf(X_test, mean=mu1_2, cov=sigma11_2)
      n = 10
      print(np.linalg.det(sigma11_2))
      #likelihood = (1/((2*np.pi)**(n/2)*np.sqrt(np.linalg.det(sigma11_2))))*np.exp(-0.5*(X_test-mu1_2).T @ np.linalg.inv(sigma11_2) @ (X_test-mu1_2))
      #score += np.log(likelihood)
      

      #print(multivariate_normal.pdf(X_test, mean=mu1_2, cov=sigma11_2))
    #print(score)
    #if score > score_max:
      #var_best = var
      #l_best = l
      #time_best = time
      #sigma_best = var_best * np.exp(-(X - X.T)**2/l_best**2)
      #score_max = score

print("var_best", var_best)
print("l_best", l_best)
print("time_best", time_best)
print("sigma_best", sigma_best)
print("score_max", score_max)

      #Check if manipulating X is correct, otherwise we can compute the covariance for Y

sigma

for var in [1, 4, 8, 16, 32, 64, 128]:
  for time_index in range(1, 50):
    time = time_index / 10
    l = time * 72
    kernel = var*np.exp()

range(7.2, 361, 7.2)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel

x0, y0 = (1500, 600)
flow_u = data_u[:, int(x0/3), int(y0/3)]
flow_v = data_v[:, int(x0/3), int(y0/3)]

X =np.arange(100).reshape(100,1)
print(flow_u.shape)
print(X.shape)

#kernel = V * RBF(L)

# define model
model = GaussianProcessClassifier()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, flow_u)
# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
parameters = {'kernel':[1*RBF(), 2*RBF(), 4*RBF(), 8*RBF(), 10*RBF(), 50*RBF(), 100*RBF()], 'C':[1, 10]}
# define search
#search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
#results = search.fit(X, y)

from sklearn.model_selection import KFold