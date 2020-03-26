import numpy as np 

def fc_backward(dout, cache):
	"""
	Computes the backward pass for an affine(fully-connected) layer.

	Inputs:
	- dout: Layer partial derivative wrt loss of shape(N, M) (Same as output)
	- cache: (x, w, b) inputs from previous forward computation

	N: Mini-batch size
	M: Number of outputs of fully connected layer
	D: Input dimension
	d_1, ..., d_k: Single input dimension

	Returns a tuple of:
	-dx: Gradient with respect to x, of shape(N, d1, ..., d_k)
	-dw: Gradient with respect to w, of shape(D, M)
	-db: Gradient with respect to b, of shape(M, )
	"""

	x, w, b = cache
	dx, dw, db = None, None, None

	# Get batch size (first dimension)
	N = x.shape[0]

	# Get dX (Same format as x)
	dx = np.dot(dout, w.T)
	dx = dx.reshape(x.shape)

	# Get dW (Same format as w)
	# Reshape activations to [Nx(d_1, ...., d_k)], which will be a 2d matrix 
	# [NxD]
	reshpaed_input = x.reshape(N, -1)
	# Transpose then dot product with dout
	dw = reshaped_input.T.dot(dout)

	# Get dB(Same format as b)
	db = np.sum(dout, axis = 0)

	# Return outputs
	return dx, dw, db