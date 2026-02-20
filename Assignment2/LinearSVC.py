import numpy as np
class LinearSVC:
    """LinearSVC classifier with Soft Margin and Regularization
    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        Random number generator seed for random weight
        initialization.
    
    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    b_ : Scalar
        Bias unit after fitting.
    errors_ : list
        Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y, C):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of
            examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.
        C : float, must be > 0
            margin parameter that determines soft margin size
            i.e. how much error we are willing to allow
            high C penalizes errors more heavily with tighter margin
            (potential overfitting)
            low C allows more errors with wider margin
            (potential underfitting)

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
        size=X.shape[1])
        self.b_ = np.float64(0.)
        self.class_errors_ = []
        self.margin_violations_ = []
        self.losses_ = []
        for _ in range(self.n_iter):
            class_errors = 0
            margin_violations = 0
            for xi, yi in zip(X, y):
                inside_margin = yi * (np.dot(xi, self.w_) + self.b_) >= 1
                if inside_margin:
                    self.w_ -= self.eta * self.w_
                else:
                    self.w_ -= self.eta * (self.w_ - C * yi * xi)
                    self.b_ += self.eta * C * yi
                if (self.predict(xi) != yi) : class_errors += 1
                if (self.net_input(xi) * yi < 1) : margin_violations += 1
            self.class_errors_.append(class_errors)
            self.margin_violations_.append(margin_violations)
            self.losses_.append(self.loss(X,y,C))
        return self
    
    # For SGD style, loss per datum
    def loss_single(self, Xi, yi, C) :
        hinge = max(0,1 - yi * np.dot(Xi,self.w_))
        squared_norm = 0.5 * np.dot(self.w_,self.w_)
        return squared_norm + C * hinge
    
    # For mini or full batch, avg loss over data
    def loss(self, X, y, C):
        hinge = np.sum(self.hinge_loss(X, y))
        squared_norm = 0.5 * np.dot(self.w_, self.w_)
        return squared_norm + (C * hinge / X.shape[0])
    
    # Hinge loss function
    def hinge_loss(self, X, y):
        margins = 1 - y * (X @ self.w_)
        return np.maximum(0, margins)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    # Unlie Adaline, we push to -1,1 which is standard for SVM
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)