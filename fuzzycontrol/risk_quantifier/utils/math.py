import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple

def calculate_line_equation(point1, point2)-> Tuple[float, float]:
    """
    Calculate the slope (m) and y-intercept (b) of the line passing through two points.

    Parameters:
    - point1: A tuple (x1, y1) representing the first point.
    - point2: A tuple (x2, y2) representing the second point.

    Returns:
    - A tuple (m, b) where:
        - m is the slope of the line.
        - b is the y-intercept of the line.

    Raises:
    - ValueError: If the line is vertical (i.e., x1 == x2), as the slope would be undefined.
    """
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        raise ValueError("The line is vertical; slope is undefined.")

    # Calculate the slope
    m = (y2 - y1) / (x2 - x1)

    # Calculate the y-intercept
    b = y1 - m * x1

    return m, b

def find_intersection(m1, b1, m2, b2)->np.array:
    """
    Calculate the intersection point of two lines given their slopes and y-intercepts.

    Parameters:
    - m1: Slope of the first line.
    - b1: Y-intercept of the first line.
    - m2: Slope of the second line.
    - b2: Y-intercept of the second line.

    Returns:
    - A tuple (x, y) representing the intersection point.

    Raises:
    - ValueError: If the lines are parallel (i.e., have the same slope).
    """
    if m1 == m2:
        raise ValueError("The lines are parallel and do not intersect.")

    # Calculate x coordinate of intersection
    x = (b2 - b1) / (m1 - m2)

    # Calculate y coordinate of intersection using one of the line equations
    y = m1 * x + b1

    return np.array([x, y])

def calculate_proportion(A, B, C)-> float:
    """
    Calculate the parameter t for point C along the line segment defined by points A and B.

    Parameters:
    - A: Tuple (x1, y1) representing the coordinates of point A.
    - B: Tuple (x2, y2) representing the coordinates of point B.
    - C: Tuple (x, y) representing the coordinates of point C.

    Returns:
    - t: The parameter indicating the position of C along segment AB.
    """
    x1, y1 = A
    x2, y2 = B
    x, y = C

    # Calculate t using x-coordinates if A and B are not vertical
    if x2 != x1:
        t = (x - x1) / (x2 - x1)
    # If A and B are vertical, use y-coordinates
    elif y2 != y1:
        t = (y - y1) / (y2 - y1)
    else:
        raise ValueError("Points A and B cannot be the same.")

    return t

def calculate_proportion(A, B, C):
    """
    Calculate the parameter t for point C along the line segment defined by points A and B.

    Parameters:
    - A: Tuple (x1, y1) representing the coordinates of point A.
    - B: Tuple (x2, y2) representing the coordinates of point B.
    - C: Tuple (x, y) representing the coordinates of point C.

    Returns:
    - t: The parameter indicating the position of C along segment AB.
    """
    x1, y1 = A
    x2, y2 = B
    x, y = C

    # Calculate t using x-coordinates if A and B are not vertical
    if x2 != x1:
        t = (x - x1) / (x2 - x1)
    # If A and B are vertical, use y-coordinates
    elif y2 != y1:
        t = (y - y1) / (y2 - y1)
    else:
        # raise ValueError("Points A and B cannot be the same.")
        t = 1

    return t

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