import matplotlib.pyplot as plt

import numpy as np


def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Training")
    xTrain = data[:, 0:2]
    yTrain = data[:, 2]

    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[:, 0], data[:, 1], data[:, 2], name="Validation")
    xValidate = data[:, 0:2]
    yValidate = data[:, 2]

    return xTrain, yTrain, xValidate, yValidate


def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]  # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:, :, 1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    X = generateDataFromGMM(N, gmmParameters)
    return X


def generateDataFromGMM(N, gmmParameters):
    #    Generates N vector samples from the specified mixture of Gaussians
    #    Returns samples and their component labels
    #    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors']  # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    X = np.zeros((n, N))
    labels = np.zeros((1, N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C + 1))
    thresholds[:, 0:C] = np.cumsum(priors)
    thresholds[:, C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:, l]))
        Nl = len(indl[1])
        labels[indl] = (l + 1) * 1
        u[indl] = 1.1
        X[:, indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl))

    # NOTE TRANPOSE TO GO TO SHAPE (N, n)
    return X.transpose()


def plot3(a, b, c, name="Training", mark="o", col="b"):
    # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    plt.title("{} Dataset".format(name))
    # To set the axes equal for a 3D plot
    ax.set_box_aspect((np.ptp(a), np.ptp(b), np.ptp(c)))
    plt.show()


if __name__ == '__main__':
    hw2q2()

from sys import float_info  # Threshold smallest positive floating value
import numpy as np
import matplotlib.pyplot as plt # For general plotting
import hw2q2
from math import ceil, floor 

from scipy.optimize import minimize
from scipy.stats import multivariate_normal # MVN not univariate

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(7)

plt.rc('font', size=22)          
plt.rc('axes', titlesize=18)   
plt.rc('axes', labelsize=18)     
plt.rc('xtick', labelsize=14)    
plt.rc('ytick', labelsize=14)  
plt.rc('legend', fontsize=16)    
plt.rc('figure', titlesize=22)  
X_train, y_train, X_valid, y_valid = hw2q2.hw2q2() 

print(np.shape(np.asarray(X_train)))

N_train = len(y_train)
N_valid = len(y_valid)

X_train = np.column_stack((np.ones(N_train), X_train))
X_valid = np.column_stack((np.ones(N_valid), X_valid))
def cubic_transformation(X):
    n = X.shape[1]
    phi_X = X
    
    # Take all monic polynomials for a quadratic
    X_1 = X[:, 1]
    X_2 = X[:, 2]

    phi_X = np.column_stack((phi_X, X_1 * X_1, X_1 * X_2, X_2 * X_2, X_1*X_1*X_2, X_1*X_2*X_2, X_1*X_1*X_1, X_2*X_2*X_2))

    return phi_X

# Mean Squared Error (MSE) loss
def lin_reg_loss(theta, X, y):
    # Linear regression model X * theta
    predictions = X.dot(theta)
    # Residual error (X * theta) - y
    error = predictions - y
    # Loss function is MSE
    loss_f = np.mean(error**2)

    return loss_f

def plot3(a, b, c, name="Training", mark="o", col="b"):
    # Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$y$")
    plt.title("{} Dataset".format(name))
    # To set the axes equal for a 3D plot
    ax.set_box_aspect((np.ptp(a), np.ptp(b), np.ptp(c)))
    plt.show()

def analytical_solution_mle(X, y):
    # Analytical solution is (X^T*X)^-1 * X^T * y 
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

X_train_cube = cubic_transformation(X_train)

theta_opt_mle = analytical_solution_mle(X_train_cube, y_train)
print('The theoretically optimal theta paramaters from MLE are: ', theta_opt_mle)

analytical_preds = cubic_transformation(X_train).dot(theta_opt_mle)

mse_mle = lin_reg_loss(theta_opt_mle, cubic_transformation(X_valid), y_valid)
print('The average-squared error for the ML esimator is {}'.format(mse_mle))


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train[:, 1], X_train[:, 2], analytical_preds, marker="o", color="b")
ax.scatter(X_train[:, 1], X_train[:, 2], y_train, marker='x', color='r')
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$y$")
plt.title("{} Dataset".format("ML Training"))
# To set the axes equal for a 3D plot
ax.set_box_aspect((np.ptp(X_train[:, 1]), np.ptp(X_train[:, 2]), np.ptp(y_train)))

analytical_preds = cubic_transformation(X_valid).dot(theta_opt_mle)


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_valid[:, 1], X_valid[:, 2], analytical_preds, marker="o", color="b")
ax.scatter(X_valid[:, 1], X_valid[:, 2], y_valid, marker='x', color='r')
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$y$")
plt.title("{} Dataset".format("ML Validation"))
# To set the axes equal for a 3D plot
ax.set_box_aspect((np.ptp(X_valid[:, 1]), np.ptp(X_valid[:, 2]), np.ptp(y_valid)))

def analytical_solution_map(X, y, gamma):
    n = np.size(np.asarray(X), axis=1)
    # Analytical solution is (X^T*X+gamma*I)^-1 * X^T * y 
    return np.linalg.inv(X.T.dot(X)+gamma*np.eye(n)).dot(X.T).dot(y)

theta_opt_map = analytical_solution_map(X_train_cube, y_train, 1e-4)
analytical_preds = X_train_cube.dot(theta_opt_map)
mse_map = lin_reg_loss(theta_opt_map, cubic_transformation(X_valid), y_valid)
print('The average-squared error for a MAP estimator with gamma = 1e-4 is {}'.format(mse_map))

theta_opt_map = analytical_solution_map(X_train_cube, y_train, 1e4)
analytical_preds = X_train_cube.dot(theta_opt_map)
mse_map = lin_reg_loss(theta_opt_map, cubic_transformation(X_valid), y_valid)
print('The average-squared error for a MAP estimator with gamma = 1e4 is {}'.format(mse_map))

gammas_test = np.geomspace(1e-4, 1e4, 10000)
gammas_mse = np.zeros(len(gammas_test))

index=0
for g in gammas_test:
    theta_opt_map = analytical_solution_map(X_train_cube, y_train, g)
    analytical_preds = X_train_cube.dot(theta_opt_map)
    gammas_mse[index] = lin_reg_loss(theta_opt_map, cubic_transformation(X_valid), y_valid)
    index+=1

fig, ax = plt.subplots()
plt.plot(gammas_test, gammas_mse)
ax.set_xscale('log')
plt.xlabel('Gamma values')
plt.ylabel('Mean squared error')
plt.title('MAP MSE spanning Gammas')
plt.grid(True)
plt.show


