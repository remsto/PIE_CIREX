import numpy as np

L = [(1, 2), (2, 3)]
print(np.var(L, axis=0))

avg = np.average(L, axis=0)
print(avg)
