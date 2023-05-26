import numpy as np


class Perceptron:
    def __init__(self, input_size) -> None:
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def feed_forward(self, X: np.array) -> float:
        return np.dot(X, self.weights) + self.bias[0]

    def mse_cost(self, X: np.array, y: np.array) -> np.array:
        average = np.zeros(1)
        for i in range(len(X)):
            activation = self.feed_forward(X[i]) - y[i]
            average += activation * activation

        average /= len(X)
        return average

    def mae_cost(self, X: np.array, y: np.array) -> np.array:
        average = np.zeros(1)
        for i in range(len(X)):
            average += np.absolute(self.feed_forward(X[i]) - y[i])

        average /= len(X)
        return average

    def finite_diff(self, X, y, eps, lr, cost):
        initial_cost = cost(X, y)
        dweights = np.zeros(len(self.weights))
        dbias = np.zeros(1)

        for i in range(len(self.weights)):
            self.weights[i] += eps
            new_cost = cost(X, y)
            self.weights[i] -= eps

            dweights[i] = (new_cost - initial_cost) / eps

        self.bias += eps
        new_cost = cost(X, y)
        self.bias -= eps

        dbias = (new_cost - initial_cost) / eps

        dweights *= lr
        dbias *= lr

        self.weights -= dweights
        self.bias -= dbias
