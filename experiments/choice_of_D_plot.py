import numpy as np
import matplotlib.pyplot as plt

D_values = np.linspace(1.9, 10, 500)

# Expected fraction selected at least once
selected_once = 1 - np.exp(-D_values)

# Expected fraction selected at least twice
selected_twice = 1 - np.exp(-D_values) - D_values * np.exp(-D_values)

plt.figure(figsize=(8, 5))

plt.plot(D_values, selected_once,
         label="Selected at least once")

plt.plot(D_values, selected_twice,
         label="Selected at least twice")

plt.xlabel("D", fontsize=14)
plt.ylabel("Expected fraction of stage-3 nodes", fontsize=14)
plt.grid(True)

plt.ylim(0.85, 1.01)

plt.legend()

plt.show()