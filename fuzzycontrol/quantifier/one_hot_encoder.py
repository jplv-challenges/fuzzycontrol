import numpy as np

from fuzzycontrol.quantifier import BaseQuantifier
from typing import List

class OneHotEncoder(BaseQuantifier):
    
    def __init__(self, sentiments:List[str]):
        self.sentiments = sentiments

    def __call__(self, sentiment:str)->np.array:
        """
        One-hot encodes the sentiment.
        :param sentiment: The sentiment to encode.
        :return: A list of 0s and 1s representing the one-hot encoding of the sentiment.
        """
        one_hot = np.zeros(len(self.sentiments), dtype=int)
        one_hot[self.sentiments.index(sentiment)] = 1
        return one_hot