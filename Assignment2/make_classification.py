import numpy as np
import random

def make_classification(d=2,n=100,u=1,seed=random.randint(1,1000), testSplit=0.7) :
    """
    Makes a classification dataset for ML training

    Parameters
    ----------
    d: int
        number of dimensions
    n: int
        number of data samples
    u: float
        range [-u,u] for samples and solution vector
    seed: int
        seed for random generation
    testSplit: float in (0.0,1.0)
        split for training vs testing data, i.e. 0.7 is 70% training 30% test
    
    Output
    ----------
    trainingX : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples
        is the number of examples and
        n_features is the number of features.
    trainingy : array-like, shape = [n_examples]
        Target values for training data.
    testX: {array-like}, shape = [n_examples, n_features]
        Same as trainingX but for testing
    testy: array-like, shape = [n_examples]
        same as testy but for testing
    """
    random.seed(seed)
    seed1 = random.randint(1,100000)
    seed2 = random.randint(1,100000)
    a = randomVec(d,-1*u,u,seed1)
    X = randomSamples(n,d,-1*u,u,seed2)
    y = getLabel(X,a)
    trainingRange = int(testSplit * n)
    trainingX = X[:trainingRange]
    trainingy = y[:trainingRange]
    testX = X[trainingRange:]
    testy = y[trainingRange:]
    return trainingX, trainingy, testX, testy, a

def randomVec(d=2,min=-1, max=1, seed=0) :
    random.seed(seed)
    out = np.zeros(d)
    for i in range(out.shape[0]):
        out[i] = random.uniform(min, max)
    return out

def randomSamples(n=10, d=2, min=-1,max=1, seed=0) :
    X = np.zeros((n,d))
    random.seed(seed)
    for iter in range(n) :
        for dim in range(d) :
            X[iter][dim] = random.uniform(min,max)
    return X

def getLabel(X, a) :
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]) :
        if np.dot(a,X[i]) < 0 :
            y[i] = -1
        else :
            y[i] = 1
    return y