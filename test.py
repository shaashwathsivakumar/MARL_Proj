import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

# Specify probability of success
p = 0.3  # You can change this value

# Generate x values (number of trials until first success)
x = np.arange(1, 20)  # First few values where PMF is significant

# Get PMF values for the geometric distribution
pmf_values = geom.pmf(x, p)

# Plotting
plt.figure(figsize=(8, 5))
plt.stem(x, pmf_values, basefmt=" ", use_line_collection=True)
plt.title(f'Geometric Distribution PMF (p = {p})')
plt.xlabel('Number of Trials until First Success')
plt.ylabel('Probability')
plt.grid(True)
plt.xticks(x)
plt.show()
