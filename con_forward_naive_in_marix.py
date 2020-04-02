import numpy as np 
def conv_forward_naive_in_matrix(x, w, b, conv_param):
	"""
	A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """

  out = None
  pad_num = conv_param['pad']
  stride = conv_param['stride']
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  H_prime = (H + 2 * pad_num - HH) // stride + 1
  W_prime = (W + 2 * pad_num - WW) // stride + 1
  out = np.zeros([N, F, H_prime, W_prime])

  #im2col

  for im_num in range(N):
  	im = x[im_num, :, :, :]
  	im_pad = np.pad(im, ((0,0), (pad_num, pad_num), (pad_num, pad_num)), 'constant')
  	filter_col = np.reshape(w, (F, -1))
  	mul = im_col.dot(filter_col.T) + b
  	out[im_num, :, :, :] = col2im(mul, H_prime, W_prime, 1)
  cache = (x, w, b, conv_param)