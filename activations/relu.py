from .activation import Activation
import numpy as np


class ReLU(Activation):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def feed_forward(self, X):
        for i in range(len(X[0])):
            self.output_array[0][i] = max(0.0, X[0][i])

        return self.output_array
