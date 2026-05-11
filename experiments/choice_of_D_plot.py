
import numpy as np

import matplotlib.pyplot as plt

def prob_all_nodes_selected_poisson(n, D):

    """
    Poisson approximation for coupon collector:
    P(all nodes selected) ≈ exp(-n * exp(-D))
    where number of draws is m = D*n.
    """

    return np.exp(-n * np.exp(-D))

n = 12**3

D_values = np.linspace(2, 10, 100)
probabilities = prob_all_nodes_selected_poisson(n, D_values)

plt.figure(figsize=(8, 5))
plt.plot(D_values, probabilities)
plt.xlabel("D")
plt.ylabel("P(all stage-3 nodes are selected)")
plt.title(f"Poisson approximation, n = {12}")
plt.grid(True)



import numpy as np

import matplotlib.pyplot as plt

D_values = np.linspace(2, 10, 500)

# Expected fraction of selected nodes

expected_fraction = 1 - np.exp(-D_values)

plt.figure(figsize=(8, 5))

plt.plot(D_values, expected_fraction)

plt.xlabel("D")

plt.ylabel("Expected fraction of stage-3 nodes selected")

plt.grid(True)

plt.ylim(0.85, 1.01)

plt.show()