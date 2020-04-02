import numpy as np 
def col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c):
	 """
    Args:
      dim_col: gradients for im_col,(h_prime*w_prime,hh*ww*c)
      h_prime,w_prime: height and width for the feature map
      strid: stride
      hh,ww,c: size of the filters
    Returns:
      dx: Gradients for x, (C,H,W)
    """
    H = (h_prime - 1) * stride + hh
    W = (w_prime - 1) * stride + ww
    dx = np.zeros([c, H, W])
    for i in range(h_prime * w_prime):
    	row = dim_col[i, :]
    	h_start = (i / w_prime) * stride
    	w_start = (i % w_prime) * stride
    	dx[:, h_start : h_start + hh, w_start : w_start + ww] += np.reshape(row, (c, hh, ww))
    return dx