#%%
from fuzzycontrol.aggregator import Summationer
from fuzzycontrol.quantifier import OneHotEncoder
from fuzzycontrol.rule import Percentage
from fuzzycontrol.risk_quantifier import CentroidQuantifier, TernaryQuantifier
from fuzzycontrol.risk_quantifier.utils.line import Line

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize

#%%
sentiments = ["negative", "neutral", "positive"]

one_hot_encoder = OneHotEncoder(sentiments)

coeffs = [5, 5, 2]
opinions = []
for i, coeff in enumerate(coeffs):
    opinions += coeff * [sentiments[i]]

# %%
encodings = []
for opinion in opinions:
    encodings.append(one_hot_encoder(opinion))
encodings = np.array(encodings)
print(encodings)
# %%

aggregator = Summationer()
aggreg_res = aggregator(encodings)
print(aggreg_res)
# %%

rule = Percentage()
rule_res = rule(aggreg_res)
print(rule_res)

#%%
n = len(rule_res)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
vertices = np.column_stack((rule_res * np.cos(angles), rule_res * np.sin(angles)))

max_vertices = np.column_stack((np.cos(angles),  np.sin(angles)))

# Function to compute the centroid of an array of vertices
def compute_centroid(vertices):
    return np.mean(vertices, axis=0)

centroid = compute_centroid(vertices)
max_centroid = compute_centroid(max_vertices)
print(f"Centroid: {centroid}")
print(f"Max Centroid: {max_centroid}")

# plot vertices
plt.figure(figsize=(6, 6))

# Plot the edges formed by the max_vertices
m = -1.5
b = 0.25
risk_frontier = Line(m, b)

points = np.linspace(-0.4, 1, 100)
plt.plot(points, risk_frontier(points), label='Risk Frontier', color='black')

for i in range(len(max_vertices)):
    plt.plot([max_vertices[i][0], max_vertices[(i + 1) % len(max_vertices)][0]],
             [max_vertices[i][1], max_vertices[(i + 1) % len(max_vertices)][1]], 'r-')


for i in range(vertices.shape[0]):
    plt.plot(max_vertices[i, 0], max_vertices[i, 1], 'o', color='red')
    plt.plot(vertices[i, 0], vertices[i, 1], 'o', color='blue')
    # Write a text label next to each max vertex sentiment[i]
    plt.text(max_vertices[i, 0], max_vertices[i, 1], sentiments[i], fontsize=12, ha='right')
plt.plot(centroid[0], centroid[1], 'b+', label='Centroid')
plt.plot(max_centroid[0], max_centroid[1], 'r+', label='Max Centroid')

max_risk_index = 0
max_risk_vertex = max_vertices[max_risk_index]


max_val = np.max(np.abs(max_vertices))
plt.xlim(-max_val, max_val)
plt.ylim(-max_val, max_val)
plt.show()

risk_quantifier = CentroidQuantifier(0)
print(risk_quantifier(rule_res))

#%%
m = -0.8
b = 0.6
ternary_quantifier = TernaryQuantifier({"m": m, "b": b})
risk_point = ternary_quantifier.point_in_ternary(rule_res)

ternary_points = np.array([
    [0, 0],
    [1, 0],
    [0.5, np.sqrt(3) / 2],
])

plt.figure(figsize=(6, 6))
for i in range(ternary_points.shape[0]):
    plt.plot(ternary_points[i, 0], ternary_points[i, 1], 'o', color='red')
    # Write a text label next to each max vertex sentiment[i]
    plt.text(ternary_points[i, 0], ternary_points[i, 1], sentiments[i], fontsize=12, ha='right')

for i in range(ternary_points.shape[0]):
    plt.plot([ternary_points[i][0], ternary_points[(i + 1) % len(ternary_points)][0]],
             [ternary_points[i][1], ternary_points[(i + 1) % len(ternary_points)][1]], 'r-')

x_frontier = np.linspace(0, 1, 100)
y_frontier = m * x_frontier + b
plt.plot(x_frontier, y_frontier, label='Risk Frontier', color='yellow')

plt.plot(risk_point[0], risk_point[1], 'b+', label='Risk Point')

plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.show()

print(ternary_quantifier(rule_res))
