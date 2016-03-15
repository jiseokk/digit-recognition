from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

a = np.array([[1,2], [1,0]])
b = np.array([[1,1], [1,0]])
print np.dot(a,b)
print np.sum(a, axis=0)

def computeProbabilities(X, theta):
    X_mod = np.transpose(X)
    print theta.shape, X_mod.shape
    mult = np.dot(theta,X_mod)
    print mult.shape
    expo = np.exp(mult)    
    sums = np.sum(expo, axis=0)
    print np.sop
    print expo
    
    
    #product = np.dot(theta, X)
    #expo = np.exp(product)
    #cum = np.sum(expo)
    #prob = expo/cum
    #return prob

def computeCostFunction(X, Y, theta, lambdaFactor):
    #CODE HERE
    pass

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor):
    #CODE HERE
    pass

def softmaxRegression(X, Y, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor)
    return theta, costFunctionProgression
    
def getClassification(X, theta):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def computeTestError(X, Y, theta):
    errorCount = 0.
    assignedLabels = getClassification(X, theta)
    return 1 - np.mean(assignedLabels == Y)
