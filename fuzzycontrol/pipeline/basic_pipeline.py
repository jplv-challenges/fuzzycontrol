import numpy as np
from typing import Dict, List

from fuzzycontrol.aggregator import Summationer
from fuzzycontrol.quantifier import OneHotEncoder
from fuzzycontrol.rule import Percentage
from fuzzycontrol.risk_quantifier import TernaryQuantifier
from fuzzycontrol.pipeline import BasePipeline

class BasicPipeline(BasePipeline):

    def __init__(self, sentiments:List[str], risk_frontier_params:Dict[str, float] = {
        'm': -0.8, 'b': 0.6
    }, max_risk_component:int = 0):
        
        self.encoder = OneHotEncoder(sentiments)
        self.aggregator = Summationer()
        self.rule = Percentage()
        self.risk_quantifier = TernaryQuantifier(risk_frontier_params, max_risk_component)

    def __call__(self, vote: List[str])->float:
        encodings = []
        for opinion in vote:
            encodings.append(self.encoder(opinion))
        encodings = np.array(encodings)
        aggregation = self.aggregator(encodings)
        vote_proportions = self.rule(aggregation)
        risk_percentage = self.risk_quantifier(vote_proportions)

        return risk_percentage
