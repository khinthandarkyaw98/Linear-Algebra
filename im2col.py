import numpy as np 
def im2col(x, hh, ww, stride):
	"""
    Args:
      x: image matrix to be translated into columns, (C,H,W)
      hh: filter height
      ww: filter width
      stride: stride
    Returns:
      col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
            new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    """

    c, h, w = x.shape
    new_h = (h - hh) // stride + 1
    new_w = (w - ww) // stride + 1
    col = np.zeros([new_h * new_w, c * hh * ww])

    for i in range(new_h):
    	for j in range(new_w):
    		patch = x[..., i * stride : i * stride + hh, j * stride : j * stride + ww]
    		col[i * new_w + j, :] = np.reshape(patch, -1)
    	return col