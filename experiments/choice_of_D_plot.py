import numpy as np
import matplotlib.pyplot as plt
from math import comb

def prob_all_selected(n, D):
    """
    Probability that all n nodes are selected
    after D*n draws with replacement.
    """
    m = int(round(D * n))

    # Inclusion-exclusion: P(all nodes selected)
    p_all_selected = 0.0
    for k in range(n + 1):
        p_all_selected += (-1)**k * comb(n, k) * ((n - k) / n) ** m

    return p_all_selected


# Parameters
n = 100
D_values = np.linspace(1, 10, 100)

probabilities = [prob_all_selected(n, D) for D in D_values]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(D_values, probabilities)
plt.xlabel("D")
plt.ylabel("P(all nodes are selected)")
plt.title(f"Probability of all nodes being selected, n = {n}")
plt.grid(True)
plt.show()