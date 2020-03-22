import numpy as np 
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
	N, H = prev_c.shape

	# forward pass in steps
	# step 1: calculate activation vector
	a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b.T

	# x = input
	# Wx = Weight of input
	# h = hidden
	# Wh = Weight of hidden layer

	# step 2: input gate
	a_i = sigmoid(a[:, 0, H])

	# step 3: forget gate
	a_f = sigmoid(a[:, H:2*H])

	# step 4: output gate
	a_o = sigmoid(a[:, 2*H:3*H])

	# step 5: block input gate
	a_g = np.tanh(a[:, 3*H:4*H])

	# step 6: next cell state
	next_c = a_f * prev_c + a_i * a_g

	# step 7: next hidden state
	next_h = a_o *  np.tanh(next_c)

	# we are having *.copy() since python params are pass by refeence
	cache = (x, prev_h.copy(), prev_c.copy(), a, a_i, a_f, a_o, a_g, next_h, next_c, Wx, Wh)

	return next_h, next_c, cache