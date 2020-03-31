import numpy as np 
def con_forward_naive(x, w, b, conv_param):
	""" 
	Computes the forward pass for the Convolution layer.(Naive)
	Input:
	- x : Input data of shape (N, C, H, W)
	- w : Filter weights of shape (F, C, HH, WW)
	- b : Biases, of shape (F, )
	- conv_param : A dictiorary with the following keys:
		- 'stride' : How much pixels the sliding window will travel
		- 'pad' : The number of pixels that will be used to zero-pad the input.

	N: Mini-batch size
	C: Input depth (i.e. 3 for RGB images)
	H/W : Image height/width	
	F 	: Number of filters on convolution layer (will be the output depth)
	HH/WW : Kernel Height/Width

	Returns a tuple of :
	- out : Output data, of shape (N, F, H', W') where H' and W' are given by
		H' = 1 + (H + 2 * pad - HH) / stride
		W' = 1 + (W + 2 * pad - WW) / stride
	-cache : (x, w, b, conv_param)
	"""
	out = None
	N, C, H, W = x.shape
	F, C, HH, WW = w.shape

	# Get parameter

	P = conv_param["pad"]
	S = conv_param["stride"]

	# Calculate output size, and initialize output value
	H_R = 1 + (H + 2 * P - HH) / S
	W_R = 1 + (W + 2 * P - WW) / S
	out = np.zeros((N, F, H_R, W_R))

	# Pad images with zeros on the border (Used to keep spatial information)
	x_pad = np.lib.pad(x, (0, 0), (0, 0), (P, P), (P, P), 'constant', constant_values = 0)

	# Apply the convolution
	for n in xrange(N): # For each element on batch
		for depth in xrange(F): # For each input depth
			for r in xrange(0, H, S): # Slide vertically taking stride into account
				for c in xrange(0, W, S): # Slide horizontally taking stride into account
					out[n, depth, r/S, c/S] = np.sum(x_pad[n, :, r:r+HH, c:c+WW] * w[depth, :, :, :] + b[depth])

	# Cache parameters and inputs for backpropagation and return output volume
	cache = (x, w, b, conv_param)
	return out, cache