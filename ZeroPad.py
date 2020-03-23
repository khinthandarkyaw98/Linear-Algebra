import numpy as np 

n = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print('n = \n' , n)

print(n.shape)

a = np.ones((4, 3))

print('a = \n', a)

''' pad_width  ---> the first (1,1) = top left to bottom down from it
					the second (1,1) = bottom left to top right
					'''
ans2 = np.pad(a, ((1, 1), (1, 1)), constant_values = 0)

print('ans2 = \n', ans2)

ans1 = np.pad(n, ((1, 1), (1, 1)), constant_values = 0)

print('ans1 = \n', ans1)