import numpy as np 
def rnn_step_forward(x, prev_h, Wx, Wh, b):
 # Step by step to make the backward propagation easier

 # Step 1
 xWx = np.dot(x, Wx)

 # Step 2
 phWh = np.dot(prev_h, Wh)

 # Step 3
 # Total
 affine = xWx + phWh = b.T # what is b.T ?

 # Step 4
 next_h = np.tanh(t)

 # Cache inputs, state, and weights

 cache = (x, prev_h.copy(), Wx, Wh, next_h, affine)

 return next_h, cache

''' Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  '''
