import numpy as np

class LogisticSGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.losses_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                loss = self._update_weights(xi, target)
                losses.append(loss)
            self.losses_.append(np.mean(losses))
        return self
    
    def fit_mini_batch_SGD(self, X, y, batch_size):
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        max = X.shape[0]

        for i in range(self.n_iter):
            X, y = self._shuffle(X, y)
            batch_losses = []
            for start in range(0, max, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                output = self.activation(self.net_input(X_batch))
                errors = y_batch-output

                self.w_ += self.eta * X_batch.T.dot(errors) / X_batch.shape[0]
                self.b_ += self.eta * errors.mean()
                loss = (-y_batch.dot(np.log(output)) -
                        (1 - y_batch).dot(np.log(1 - output))) / X_batch.shape[0]
                batch_losses.append(loss)
            self.losses_.append(np.mean(batch_losses))
        return self

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(0.0, 0.01, size=m)
        self.b_ = np.float64(0.)

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _update_weights(self, xi, target):
        z = self.net_input(xi)
        output = self.activation(z)
        error = target - output
        self.w_ += self.eta * error * xi
        self.b_ += self.eta * error
        loss = -target*np.log(output) - (1-target)*np.log(1-output)
        return loss

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
