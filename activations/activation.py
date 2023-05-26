import numpy as np


class Activation:
    def __init__(self, shape: tuple[int, int]) -> None:
        assert isinstance(shape, tuple)
        self.output_array = np.zeros(shape=shape)

    def feed_forward(self, X):
        ### Compute activation ###

        return self.output_array
