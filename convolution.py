import numpy as np 

x = np.array([0, 1, 2, 3, 4])
w = np.array([1, -1, 2])

answer = np.convolve(x, w)

# only inside the window
valid = np.convolve(x, w, 'valid')

print("After convolution : ", answer)

print("Vaid answer: ", valid)