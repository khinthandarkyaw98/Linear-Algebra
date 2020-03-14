import numpy as np 

# scalar
a = 24

# vector
b = np.array([2, -8, 7])

# Two Dimensional Array >>> Big difference from Matlab is [[]]
c = np.array([[-6], [-4], [27]])

#Matrix
d = np.array([[6, 4, 24],[1, -9, 8]])

# Matlab d(2, 3)
print(d[1,2])

print(b[2])

# size(D)
print(d.shape)