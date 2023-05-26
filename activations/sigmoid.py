from .activation import Activation
import math


class Sigmoid(Activation):
    def __init__(self, shape: tuple[int, int]) -> None:
        super().__init__(shape)

    def feed_forward(self, X):
        for i in range(len(X[0])):
            self.output_array[0][i] = 1 / (1 + math.exp(-X[0][i]))

        return self.output_array
