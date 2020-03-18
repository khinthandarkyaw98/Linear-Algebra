import numpy as np 

# 3x4 matrix
W = np.array([[0.2, -0.5, 0.1, 2], [1.5, 1.3, 2.1, 0], [0, 0.25, 0.2, 0.3]])
print('W\n', W)
print("W =", W.shape)

# 4x1 matrix
X = np.array([[56], [231], [24], [2]])
print("X =",X.shape)

# 3x1 matrix
B = np.array([[1.1], [3.2], [-1.2]])

scores = W.dot(X) + B

print(scores)
print("scores = ", scores.shape)