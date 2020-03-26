""" Computes the backward pass for ReLu
Input : 
- dout: Upstream derivatives, of any shape
- cache: Previous input (used on forward propagation)

Returns:
- dx: Gradient with respect to x
"""
# Initialize dx with None and x with cache
def relu_backward(dout, cache):
	
	dx, x = None, cache

	# Make all positive elements in x equal to dout while all the other elements
	# Become zero
	dx = dout * (x >= 0)

	# Return dx (gradient with respect to x)
	return dx