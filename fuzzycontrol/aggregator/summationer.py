import numpy as np

from fuzzycontrol.aggregator import BaseAgreggator
from typing import List

class Summationer(BaseAgreggator):
    """
    Summationer class that inherits from BaseAgreggator.
    This class is used to perform summation on a list of values.
    """

    def __call__(self, values: np.array) -> np.array:
        """
        Summationer function that takes a list of values and returns their sum.
        :param values: A list of values to be summed.
        :return: The sum of the values.
        """
        return np.sum(values, axis=0)