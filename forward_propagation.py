import numpy as np 

def fc_forward(x, w, b):
	"""
	Computes the forward pass for an affine(fully-connected) layer.

	Inputs:
	- x: Input Tensor(N, d_1, ..., d_k)
	- w: Weights(D, M)
	- b: Bias (M,)

	N: Mini-batch size
	M: Number of outputs of fully connected layer
	D: Input Dimension

	Returns a tuple of:
	-out : output, of shape(N, M)
	-cache : (x, w, b)
	"""

	out = none

	# Get batch size(first dimension)
	N = x.shape[0]

	# Reshape activations to [Nx(d_1, ..., d_k)], which will be a 2D matrix
	# [NxD]
	reshaped_input = x.reshape(N, -1)

	# Calculate Output
	out = np.dot(reshaped_input, w) + b.T

	#Save inputs for backward propagation
	cache = (x, w, b)
	return out, cache