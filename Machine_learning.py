import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import minimize
import scipy.io as sio
from scipy.stats import multivariate_normal

def estimate_gaussian(X):
  mu = np.mean(X,axis=0)
  sigma = np.std(X,axis=0)
  return mu, sigma


def select_threshold(ycv, pcv):
    max_p = np.max(pcv)
    min_p = np.min(pcv)
    step_p = (max_p - min_p) / 1400
    y_t = np.zeros(len(pcv))
    F1_main = epxilon_main = 0
    for i in np.arange(min_p, max_p, step_p):
        TP = FN = FP = 0
        for j in range(len(y_t)):
            if pcv[j] < i:
                y_t[j] = 1
            else:
                y_t[j] = 0

        for k in range(len(y_t)):
            y_predict = y_t[k]
            y_lable = ycv[k]
            if (y_lable == 1) & (y_predict == 1):
                TP += 1
            elif (y_lable == 0) & (y_predict == 1):
                FP += 1
            elif (y_lable == 1) & (y_predict == 0):
                FN += 1
        if (TP == 0):
            continue
        F1 = 2 * (TP / (TP + FP)) * (TP / (TP + FN)) / ((TP / (TP + FP)) + (TP / (TP + FN)))
        if F1 > F1_main:
            F1_main = F1
            epxilon_main = i
    return F1_main, epxilon_main


def normalize_ratings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros_like(Y)
    print(1, Ymean.shape)
    Ymean = np.mean(Y, axis=0)
    print(2, Ymean.shape)
    return Ymean, Ynorm


def cofi_cost(params, Y, R, num_users, num_movies, num_features, _lambda):
    """
    Collaborative filtering cost function
    """
    # unpack values
    X = params[:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features:].reshape((num_users, num_features))
    value1 = np.sum((np.dot(X,Theta.T)-Y)*R)
    value2 = (_lambda/2)*np.sum(Theta[:]**2)
    value3 = (_lambda/2)*np.sum(X[:]**2)
    return value1+value2+value3


def cofi_gradient(params, Y, R, num_users, num_movies, num_features, _lambda):
    X = params[:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features:].reshape((num_users, num_features))
    alpha = 0.0001
    X = X - alpha*((np.dot(((np.dot(X,Theta.T)-Y)*R),Theta)+_lambda*X))
    Theta = Theta - alpha*((np.dot(((np.dot(X,Theta.T)-Y)*R).T,X)+_lambda*Theta))
    return np.hstack((X.flatten(),  Theta.flatten()))

# Load data
data = sio.loadmat('ex8data2.mat')
X = data["X"]
Xcv = data["Xval"]
ycv = data["yval"].flatten()

# Fit mu and sigma
mu, sigma2 = estimate_gaussian(X)

# Choose epsilon
pcv = multivariate_normal.pdf(x=Xcv, mean=mu, cov=sigma2)
select_threshold(ycv, pcv)

data = sio.loadmat('ex8_movies.mat')
Y = data["Y"]
R = data["R"]
Y.shape, R.shape

print('Average rating for movie 1: {}'.format(np.mean(Y[0, R[0, :]])))

data = sio.loadmat('ex8_movieParams.mat')
Theta = data["Theta"]
X = data["X"]

# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X_ = X[:num_movies, :num_features]
Theta_ = Theta[:num_users, :num_features]
Y_ = Y[:num_movies, :num_users]
R_ = R[:num_movies, :num_users]

J = cofi_cost(np.hstack((X_.flatten(), Theta_.flatten())), Y_, R_, num_users, num_movies, num_features, 0)
print('Cost at loaded parameters: {}'.format(J))
J = cofi_cost(np.hstack((X_.flatten(), Theta_.flatten())), Y_, R_, num_users, num_movies, num_features, 1.5)
print('Cost at loaded parameters: {}'.format(J))

from scipy.optimize import fmin_cg

# Perform normalization
Ymean, Ynorm = normalize_ratings(Y, R)

num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
X = np.random.random((num_movies, num_features))
Theta = np.random.random((num_users, num_features))
initial_parameters = np.hstack((X.flatten(),  Theta.flatten()))

# Train
print("Train")
_lambda = 10
results = fmin_cg(cofi_cost, initial_parameters, args = (Y, R, num_users, num_movies, num_features, _lambda),
                  fprime = cofi_gradient)