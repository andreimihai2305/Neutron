from .activation import Activation
import math


class Tanh(Activation):
    def __init__(self, shape) -> None:
        super().__init__(shape)

    def feed_forward(self, X):
        for i in range(len(X[0])):
            nr = X[0][i]
            self.output_array[0][i] = (math.exp(nr) - math.exp(-nr)) / (
                math.exp(nr) + math.exp(-nr)
            )
        return self.output_array
