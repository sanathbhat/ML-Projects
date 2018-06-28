import pandas as pd
import numpy as np
from math import e
import seaborn as sb
import matplotlib.pyplot as plt

EPSILON = 0.00000001

def sigm(X):
    return 1/(1+e**-X)

def logistic_regression(X, y, alpha, iters=None):
    m = X.shape[0]  # number of data points
    # augment X with additional feature of all 1s
    X = np.concatenate((np.ones((m, 1)), X), 1)
    theta = np.zeros((X.shape[1], 1))

    # uses batch gradient descent
    if iters:
        for i in range(iters):
            theta = theta - alpha/m * np.dot(X.T, sigm(np.dot(X, theta))-y)
    else:
        thetaP = theta + 1
        iters = 0
        while abs(min(thetaP-theta)) > EPSILON:
            thetaP = theta
            theta = theta - alpha / m * np.dot(X.T, sigm(np.dot(X, theta)) - y)
            # print(theta.T)
            iters += 1
        print('Finished GD in {} iterations'.format(iters))
        # print('thetaP = {}\ntheta={}\nmin diff={}'.format(thetaP.T, theta.T, min(thetaP-theta)))
    return theta


if __name__=='__main__':
    datadf = pd.read_csv('sampledata/studenttest.txt', header=None, names=['f0','f1','accepted'])
    data = datadf.as_matrix()
    X = data[:, :-1]
    y = data[:, [-1]]
    theta = logistic_regression(X, y, 0.001)
    print(theta)

    #plot the data points
    sb.lmplot(x='f0', y='f1', data=datadf, hue='accepted', fit_reg=False)

    #plot a boundary contour based on theta
    # cx = np.linspace(-1, 1.25, 226)
    # cy = np.linspace(-1, 1.25, 226)
    cx = np.linspace(start=0, stop=100, num=101)
    cy = np.linspace(start=0, stop=100, num=101)

    # for i in cx:
    #     for j in cy:
    # print(theta[0])
    # print(theta[1])

    # boundary = [sigm(theta[0] + theta[1] * i + theta[2] * j) for j in cy for i in cx]
    # print(boundary)
    boundary = [[i, j] for j in cy for i in cx if abs(sigm(theta[0]+theta[1]*i+theta[2]*j)-0.5) < 0.02]
    bdf = pd.DataFrame(boundary, columns=['x', 'y'])

    sb.lmplot(x='x', y='y', data=bdf, fit_reg=False, scatter_kws={"s": 1})
    plt.show()
