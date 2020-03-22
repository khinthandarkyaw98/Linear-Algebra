import numpy as np 
def lstm_step_backward(dnext_h, dnext_c, cache):
	(x, prev_h, prev_c, a, a_i, a_f, a_o, a_g, next_h, next_c, Wx, Wh) = cache

	N, H = dnext_h.shape
	da = np.zeros(a.shape)

	# step 7:
	dnext_c = dnext_c.copy()
	dnext_c += dnext_h * a_o * (1 - np.tanh(next_c) ** 2)
	da_o = np.tanh(next_c) * dnext_h

	# step 6:
	da_f = dnext_c * prev_c
	dprev_c = dnext_c * a_f
	da_i = dnext_c * a_g
	da_g = dnext_c * a_i

	# step 5:
	da[:, 3*H:4*H] = (1 - np.square(a_g)) * da_g

	# step 4:
	da[:, 2*H:3*H] = (1 - a_o) * a_o * da_o

	# step 3:
	da[:, H:2*H] = (1 - a_f) * a_f * da_f

	# step 2:
	da[:, 0:H] =(1 - a_i) * a_i * da_i

	# step 1:
	db = np.sum(da, axis = 0)
	dx = da.dot(Wx.T)
	dWx = x.T.dot(da)
	dprev_h = da.dot(Wh.T)
	dWh = prev_h.T.dot(da)

	return dx, dprev_h, dprev_c, dWx, dWh, db
 