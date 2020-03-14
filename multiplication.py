# Matrix multplication

import numpy as np 

# 3x3 Matrix

weight = np.array([[0.2, -0.5, 0.1, 2, 1.1], [1.5, 1.3, 2.1, 0, 3.2], [0, 0.25, 0.2, -0.3, -1.2]])

x = np.array([56, 231, 24, 2, 1])

ans = weight.dot(x)

print(ans)
