import numpy as np
def dropout_backward(dout, cache):

	""" 
	Perform the backward pass for (inverted) dropout.
	Inputs:
	-dout: Upstream deriavtives, of any shape
	-cache: (dropout_param, mask) from dropout_forward

	"""

	# Recover dropout parameters (p, mask, mode) from cache
	dropout_param, mask = cache
	mode = dropout_param['mode']

	dx = None
	# Back propagate (Dropout layer has no parameters just input X)
	if mode == 'train':
		# Just back propagate dout from the neurons that were used during dropout
		dx = dout * mask

	elif mode = 'test':
		# Disable dropout during prediction/ test
		dx = dout


	# Return dx
	return dx