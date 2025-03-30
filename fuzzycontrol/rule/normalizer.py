import numpy as np

from fuzzycontrol.rule import BaseRule

class Normalizer(BaseRule):
    def __call__(self, aggregate:np.array)->np.array:
        """
        Normalize the input array to the range [0, 1].

        Parameters
        ----------
        aggregate : np.array
            The input array to be normalized.

        Returns
        -------
        np.array
            The normalized array.
        """
        return (aggregate - np.min(aggregate)) / (np.max(aggregate) - np.min(aggregate))