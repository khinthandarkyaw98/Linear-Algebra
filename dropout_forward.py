import numpy as np 

def dropout_forward(x, dropout_param):
	"""
	Performs the forward pass for (inverted) dropout.
	Inputs:
	-x: Input Data, of any shape
	- dropout_param: A dictionary wiht the following keys: (p, test/ train, seed)
	Outpurs: (out, cache)
	"""

	# Get the current dropout made, p, and seed
	p, mode = dropout_param['p'], dropout_param['mode']
	if 'seed' in dropout_param:
		# np.random.seed(0) makes the random numbers predictable
		"""
		With the seed reset (every time), the same set of numbers will appear every time.

If the random seed is not reset, different numbers appear with every invocation:
		"""
		np.random.seed(dropout_param['seed'])

		# Initialization of outpurs and mask
		mask = None
		out = None

		if mode == 'train':
			# Create an apply mask (normally p = 0.5 for half of neurns), we scale all 
			# by p to avoid having to multiply by p on backpropagation, this is called
			# inverted dropout
			mask = (np.random.rand(*x.shape) < p) / p
			# Apply mask
			out = x* mask
		elif mode == 'test':
			# During prediction no mask is used
			mask = None
			out = x

		# Save mask and dropout parameters for backpropagaiton
		cache = (dropout_param, mask)

		# Convert 'out' type and return output and cache
		out = out.astype(x.dtype, copy = False)
		return out, cache

