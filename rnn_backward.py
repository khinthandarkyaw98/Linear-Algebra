import numpy as np 
def rnn_step_backward(dnext_h, cache):
	(x, prev_h, Wx, Wh, next_h, affine) = cache

	# backward in step 
	# step 4
	# dt delta of total
	# Gradient of tanh times dnext_h

	dt = (1 - np.square(np.tanh(affine))) * (dnext_h)

	# step 3
	# Gradient of sum block
	dxWx = dt
	dphWh = dt
	db = np.sum(dt, axis = 0)

	# step 2
	# Gradient of the mul block
	dWh = prev_h.T.dot(dphWh)
	dprev_h = Wh.dot(dphWh.T).T

	# step 1
	# Gradient of the mul block
	dx = dxWx.dot(Wx.T)
	dWx = x.T.dot(dxWx)

	return dx, dprev_h, dWx, dWh, db
