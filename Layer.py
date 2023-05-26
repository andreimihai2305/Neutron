import numpy as np


class DenseLayer:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.rand(input_size, output_size)
        self.biases = np.random.rand(1, output_size)

    def feed_forward(self, X):
        return np.dot(X, self.weights) + self.biases
