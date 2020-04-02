import numpy as np 

def conv_backward_naive_in_matrix(dout, cache):
	"""
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """

  dx, dw, db = None, None, None

  x, w, b, conv_param = cache
  pad_num = conv_param['pad']
  stride = conv_param['stride']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H_prime = (H + 2 * pad_num - HH) // stride + 1
  W_prime = (W + 2 * pad_num - WW) // stride + 1

  dw = np.zeros(w.shape)
  dx = np.zeros(x.shape)
  db = np.zeros(b.shape)

  # We could calculate the bias by just summing over the right dimensions
  # Bias gradient (Sum on dout dimensions (batch, rows, cols)
  #db = np.sum(dout, axis=(0, 2, 3))

  for i in range(N):
  	im = x[i, :, :, :]
  	im_pad = np.pad(im, ((0, 0), (pad_num, pad_num), (pad_num, pad_num)), 'constant')
  	im_col = im2col(im_pad, HH, WW, stride)
  	filter_col = np.reshape(w, (F, -1)).T

  	dout_i = dout[i, :, :, :]
  	dbias_sum = np.reshape(dout_i, (F, -1))
  	dbias_sum = dbias_sum.T

  	# bias_sum = mul + b
  	db += np.sum(dbias_sum, axis = 0)
  	dmul = dbias_sum

  	# mul = im_col * filter_col
  	dfilter_col = (im_col.T).dot(dmul)
  	dim_col = dmul.dot(filter_col.T)

  	dx_padded = col2im_back(dim_col, H_prime, W_prime, stride, HH, WW, C)
  	dx[i, :, :, :] = dx_padded[:, pad_num: H + pad_num, pad_num: W + pad_num]
  	dw += np.reshape(dfilter_col.T, (F, C, HH, WW))
 return dx, dw, db