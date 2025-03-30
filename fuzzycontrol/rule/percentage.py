import numpy as np

from fuzzycontrol.rule import BaseRule

class Percentage(BaseRule):
    def __call__(self, aggregate:np.array)->np.array:
        """
        Calculate the percentage of each element in the input array.

        Parameters
        ----------
        aggregate : np.array
            The input array to calculate the percentage for.

        Returns
        -------
        np.array
            The percentage of each element in the input array.
        """
        return aggregate / np.sum(aggregate)