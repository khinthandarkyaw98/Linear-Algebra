import numpy as np 

a = np.ones((2,3))

print("Origianl a \n", a)

a_reshape = a.reshape(6, -1)

print("Reshape a \n", a_reshape)

b = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])

print(" Origianl b \n", b)

b_reshape = b.reshape(9, -1)

print(" Reshape b \n", b_reshape)