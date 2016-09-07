"""Softmax function Implementation to return the probability of the given scores"""

scores = [3.0, 1.0, 0.2]

import numpy as np
import math
def softmax(x):
    prob = []
    sigma = 0
    for i in x:
        sigma = sigma+np.exp(i)
    
    for i in x:
        prob.append((np.exp(i)/sigma))
    return np.array(prob)
    


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
