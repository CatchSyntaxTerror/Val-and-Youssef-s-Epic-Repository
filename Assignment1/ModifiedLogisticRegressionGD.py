import numpy as np

class ModifiedLogisticRegressionGD:
    """Gradient descent-based logistic regression classifier
    with the bias term absorbed into the weight vector.

    An additional feature of constant value 1 is appended to
    each input sample, and the corresponding weight functions
    as the bias. This formulation is mathematically equivalent
    to the standard logistic regression model with an explicit
    bias term.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Number of passes over the training dataset.
    random_state : int
        Seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weight vector after training, including the bias weight.
    losses_ : list
        Log-loss values for each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Append a column of ones to the array
        """
        rows = X.shape[0]
        X = np.hstack((X, np.ones((rows,1))))
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the
        number of examples and n_features is the
        number of features.
        y : array-like, shape = [n_examples]
        Target values.
        Returns
        -------
        self : Instance of ModifiedLogisticRegressionGD
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.losses_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
            loss = (-y.dot(np.log(output)) - (1 - y).dot(np.log(1 - output))) / X.shape[0]
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_)
    
    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        rows = X.shape[0]
        X = np.hstack((X, np.ones((rows,1))))
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)