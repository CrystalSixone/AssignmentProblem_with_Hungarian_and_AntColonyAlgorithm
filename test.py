import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([1,2,6])
result = [False for c in b if c not in a]
print(result)