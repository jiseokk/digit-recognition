from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

#a = np.array([[1,2,2], [2,4,1]])
#print 0.5*a
#b = np.array([[1,1], [1,0]])
#print np.dot(a,b)
#print np.sum(a, axis=0)

#X = np.array([[1,2], [1,0]])
#print augmentFeatureVector(X)

""" Testing for computeProbabilities
Y = np.array([[0,1,2], [3,4,5], [6,7,8], [9,10,11]])
sub = np.array([0,1,2,3])
mult_mod = Y - sub[np.newaxis].transpose()
print mult_mod
expo = np.exp(mult_mod)
print expo
sums = np.sum(expo, axis=1)
print sums
print sums[:,None]
here = expo / sums[:,None]
print here
"""

def computeProbabilities(X, theta):
    #print X.shape, theta.shape
    theta_mod = np.transpose(theta)
    mult = np.dot(X, theta_mod)
    #print mult
    maxima = np.amax(mult, axis=1)
    maxima_colMod = maxima[np.newaxis].transpose()
    mult_mod = mult - maxima_colMod
    expo = np.exp(mult_mod) 
    sums = np.sum(expo, axis=1)
    here = expo / sums[:,None]
    #print here.transpose()
    return here.transpose()
    
    
    
    """X_mod = np.transpose(X)
    print theta.shape, X_mod.shape
    mult = np.dot(theta,X_mod)
    print mult.shape
    expo = np.exp(mult)    
    sums = np.sum(expo, axis=0)
    print np.sop
    print expo"""
    
    
    #product = np.dot(theta, X)
    #expo = np.exp(product)
    #cum = np.sum(expo)
    #prob = expo/cum
    #return prob

def computeCostFunction(X, Y, theta, lambdaFactor):
    prob = computeProbabilities(X,theta)
    count = 0    
    for i in range(len(prob)):
        count += np.log(prob[Y[i]][i])   
    count = count/float(len(X))
    return -1*count + 0.5*lambdaFactor*np.linalg.norm(theta)**2

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor):
    m = float(len(X))    
    prob = computeProbabilities(X,theta)
    ymat = np.zeros([prob.shape[0], prob.shape[1]])
    for i in range(len(Y)):
        ymat[Y[i]][i] = 1
    new = ymat - prob
    grad_pre = (-1./m)*np.dot(new, X)
    grad = grad_pre + lambdaFactor*theta
    new_theta = theta - alpha*grad
    return new_theta
    

def softmaxRegression(X, Y, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    it = 0
    for i in range(numIterations):
        it += 1
        print it
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