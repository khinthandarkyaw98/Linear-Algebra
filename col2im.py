import numpy as np 
def col2im(mul, h_prime, w_prime, C):
	"""
      Args:
      mul: (h_prime*w_prime*w,F) matrix, each col should be reshaped to C*h_prime*w_prime when C>0, or h_prime*w_prime when C = 0
      h_prime: reshaped filter height
      w_prime: reshaped filter width
      C: reshaped filter channel, if 0, reshape the filter to 2D, Otherwise reshape it to 3D
    Returns:
      if C == 0: (F,h_prime,w_prime) matrix
      Otherwise: (F,C,h_prime,w_prime) matrix
    """
    F = mul.shape[1]
    if (C ==1):
    	out = np.zeros([F, h_prime, w_prime])
    	for i in range(F):
    		col = mul[:, i]
    		out[i, :, :] = np.reshape(col, (h_prime, w_prime,))
    else:
    	out = np.zeros([F, C, h_prime, w_prime])
    	for i in range(F):
    		col = mul[:, i]
    		out[i, :, :] = np.reshape(col, (C, h_prime, w_prime))

    return out