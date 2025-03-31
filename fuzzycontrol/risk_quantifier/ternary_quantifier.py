from fuzzycontrol.risk_quantifier import BaseRiskQuantifier
from fuzzycontrol.risk_quantifier.utils.line import Line
from fuzzycontrol.risk_quantifier.utils.math import calculate_proportion, calculate_line_equation, find_intersection

import numpy as np

class TernaryQuantifier(BaseRiskQuantifier):

    TERNARY_POINTS = np.array([
        [0, 0],
        [1, 0],
        [0.5, np.sqrt(3)/2],
    ])

    def __init__(self, risk_frontier_params: dict, max_risk_component: int = 0):
        if "m" not in risk_frontier_params or "b" not in risk_frontier_params:
            raise ValueError("risk_frontier_params must contain 'm' and 'b' keys to define the risk line")

        self.risk_frontier = Line(
            m=risk_frontier_params["m"],
            b=risk_frontier_params["b"])

        self.max_risk_component = max_risk_component


    def __call__(self, risk_vector:np.array) -> float:
        """
        Calculate the ternary quantifier for a given risk vector.

        Args:
            risk_vector (np.array): The risk vector to be evaluated.

        Returns:
            float: The ternary quantifier value.
        """

        if np.isnan(risk_vector).any():
                # Risk vector contains NaN values, one of the models in the monitor is not responding
                return 1

        if len(risk_vector) != 3:
            raise ValueError("Risk vector must have exactly 3 elements to be used with the ternary quantifier.")
            
        

        risk_point = self.point_in_ternary(risk_vector)

        if risk_point[0] == self.TERNARY_POINTS[self.max_risk_component][0] and risk_point[1] == self.TERNARY_POINTS[self.max_risk_component][1]:
            # Risk point matches with max risk point
            return 1

        m, b = calculate_line_equation(
            np.array([0, 0]),
            risk_point
        )
        risk_frontier_intersection = find_intersection(
            self.risk_frontier.m,
            self.risk_frontier.b,
            m,
            b
        )
        proportion = calculate_proportion(
            risk_frontier_intersection,
            np.array([0, 0]),
            risk_point
        )
        proportion = np.clip(proportion, 0, 1)
        return proportion

    @staticmethod
    def point_in_ternary(proportions:np.array) -> np.array:
        # https://en.wikipedia.org/wiki/Ternary_plot#Plotting_a_ternary_plot
        a, b, c = proportions
        x = 0.5 * (2*b + c) / (a + b + c)
        y = (np.sqrt(3)/2) * c / (a + b + c)
        return np.array([x, y])
