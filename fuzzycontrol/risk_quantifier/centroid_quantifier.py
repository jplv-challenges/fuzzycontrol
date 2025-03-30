import numpy as np

from fuzzycontrol.risk_quantifier import BaseRiskQuantifier

class CentroidQuantifier(BaseRiskQuantifier):

    def __init__(self, risk_coordinate: callable):
        self.risk_coordinate = risk_coordinate

    def __call__(self, risk_vector:np.array)-> float:
        n = len(risk_vector)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vertices = np.column_stack((risk_vector * np.cos(angles), risk_vector * np.sin(angles)))
        centroid = np.mean(vertices, axis=0)

        max_risk_vertex_angle = self.risk_coordinate * 2 * np.pi
        max_risk_vertex = np.array([np.cos(max_risk_vertex_angle), np.sin(max_risk_vertex_angle)])
        print(max_risk_vertex_angle)
        print(max_risk_vertex)
        print(centroid)
        print(risk_vector)

        distance = np.linalg.norm(centroid - max_risk_vertex)
        return distance

