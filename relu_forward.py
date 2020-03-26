import numpy as np 

""" 
Computes the forward pass for RELU
Input :
-x : Inputs , of any shape

Returns a tuple of : (out, cache)
The shape on the output is the same as the input
"""

def relu_forward(x):

	out = None

	# Create a function that recevies x and return x if x is bigger
	# than zero or zero if x is negative
	relu = lambda x: x * (x > 0).astype(float)
	out = relu(x)

	# Cache input and return outputs
	cahe = x
	return out, cache
	