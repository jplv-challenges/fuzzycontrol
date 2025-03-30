#%%
from fuzzycontrol.aggregator import Summationer
from fuzzycontrol.quantifier import OneHotEncoder
from fuzzycontrol.rule import Percentage
from fuzzycontrol.risk_quantifier import CentroidQuantifier

from matplotlib import pyplot as plt
import numpy as np

#%%
sentiments = ["negative", "neutral", "positive"]

one_hot_encoder = OneHotEncoder(sentiments)

coeffs = [10, 0, 0]
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

def risk_frontier(x):
    return -1.5*x+0.5

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

# Put the xlim and ylim to be the same and equal to the max of the vertices
max_val = np.max(np.abs(max_vertices))
plt.xlim(-max_val, max_val)
plt.ylim(-max_val, max_val)


plt.show()

#%%
risk_quantifier = CentroidQuantifier(0)
print(risk_quantifier(rule_res))