import numpy as np

class LogisticMiniBatchSGD:
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
        
    def fit(self, X, y, batch_size=32):
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
        pars, las, ypars, ylas = self.parse_arr(X, y, batch_size)
        parsed = np.array(pars)
        last = np.array(las)
        yparsed = np.array(ypars)
        ylast = np.array(ylas)
        for i in range(self.n_iter):
            # for average 
            self.batch_losses_ = []
            for j in range(X.shape[0] // batch_size):
                net_input = self.net_input(parsed[j])
                output = self.activation(net_input)
                # numerical stability fix
                output = np.clip(output, 1e-10, 1 - 1e-10)

                errors = (yparsed[j] - output)
                self.w_ += self.eta * parsed[j].T.dot(errors) / parsed[j].shape[0]
                loss = (-yparsed[j].dot(np.log(output)) - (1 - yparsed[j]).dot(np.log(1 - output))) / parsed[j].shape[0]
                # for average 
                self.batch_losses_.append(loss)
                # elf.losses_.append(loss)
                
            if last[0].size != 0 :
                # last batch
                net_input = self.net_input(last[0])
                output = self.activation(net_input)
                # numerical stability fix
                output = np.clip(output, 1e-10, 1 - 1e-10)

                errors = (ylast[0] - output)
                self.w_ += self.eta * last[0].T.dot(errors) / last[0].shape[0]
                loss = (-ylast[0].dot(np.log(output)) - (1 - ylast[0]).dot(np.log(1 - output))) / last[0].shape[0]
                # for average 
                self.batch_losses_.append(loss)
                # self.losses_.append(loss)
                
            # for average : 
            self.losses_.append(sum(self.batch_losses_)/len(self.batch_losses_))
            
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
    
    def parse_arr(self, X, y, batch_size):
        """
        Parses input matrix into batches
        Assumes X is 2d
        """
        i = 0
        parsed = [] 
        last_batch = []
        y_parsed = []
        y_last = []
        
        while i + batch_size < X.shape[0]:
            y_parsed.append(y[i:i+batch_size])
            parsed.append(X[i:i+batch_size])
            i += batch_size
        last_batch.append(X[i:])
        y_last.append(y[i:])
        return parsed, last_batch, y_parsed, y_last 
        