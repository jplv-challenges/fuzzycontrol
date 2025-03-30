import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple

def distance_to_curve(x: float, point: np.array, risk_frontier: Callable) -> float:
    """
    Calculate the distance from a point to a point on the curve.
    
    Args:
        x: x-coordinate on the curve
        point: point to measure distance from
        risk_frontier: function that defines the curve
        
    Returns:
        float: distance between the point and curve point
    """
    curve_point = np.array([x, risk_frontier(x)])
    return np.linalg.norm(curve_point - point)

def get_closest_point(point: np.array, risk_frontier: Callable) -> Tuple[np.array, float]:
    """
    Find the closest point on the curve to a given point.
    
    Args:
        point: point to find closest curve point to
        risk_frontier: function that defines the curve
        
    Returns:
        Tuple containing:
        - np.array: closest point on the curve
        - float: distance to the closest point
    """
    result = minimize(distance_to_curve, point[0], args=(point, risk_frontier))
    closest_x = result.x[0]
    closest_point = np.array([closest_x, risk_frontier(closest_x)])
    return closest_point, np.linalg.norm(closest_point - point)