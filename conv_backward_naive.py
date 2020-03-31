import numpy as np 

def conv_backward_naive(dout, cache):
	"""
	Computes the backward pass for the Convolution layer. (Naive)
	Inputs :
	- dout : Upstream derivatives.
	- cache : A tuple of (x, w, b, conv_param) as in conv_forward_naive
	Returns a tuple of : (dw, dx, db) gradients
	"""

	dx, dw, db = None, None, None	
	x, w, b, conv_param = cache
	N, F, H_R, W_R = dout.shape
	F, C, HH, WW = w.shape
	P = conv_param["pad"]
	S = conv_param["stride"]

	# Do zero padding on x_pad
	x_pad = np.lib.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), 'constant', constant_values = 0)

	# Initialize outputs
	dx = np.zeros(x.pad.shape)
	dw = np.zeros(w.shape)
	db = np.zeros(b.shape)

	# Calculate dx, with 2 extra col/row will be deleted
	for n in xrange(N): # For each element on batch
		for depth in xrange(F): # For each filter
			for r in xrange(0, H, S): # Slide vertically taking stride into account
				for c in xrange(0, W, S): # Slide horizontally taking stride into account
					dx[n, :, r:r+HH, c:c+WW] += dout[n, depth, r/S, c/S] * w[depth, :, :, :]

	# deleting padded rows to match real dx
	delete_rows		= range(P) + range(H+P, H+2 * P, 1)
	delete_columns 	= range(P) + range(W+P, W+2 * P, 1)
	# np.delete reurns a new array along an axis deleted
	# np.delete(arr, obj, axis = None)
	dx = np.delete(dx, delete_rows, axis = 2) # height
	dx = np.delete(dx, delete_columns, axis = 3) # width

	# Calculate dw
	"""
	range() returns the list,
	xrange() returns the xrange object
	"""
	for n in xrange(N): # For each element on batch
		for depth in xrange(F): # For each filter
			for r in xrange(H_R): # Slide vertically taking stride into account
				for c in xrange(W_R): # Slide horizontally taking stride into account
					dw[depth, :, :, :] += dout[n, depth, r, c] * x_pad[n, :, r * S : r * S + HH , c * S : c * S + WW]

	# Calculate db, 1 scalar bias per filter, so it's just a matter of summing
	# all elements of dout per filter
	for depth in range(F):
		db[depth] = np.sum(dout[:, depth, :, :])

	return dx, dw, db