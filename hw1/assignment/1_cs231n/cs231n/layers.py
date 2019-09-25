import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = x.reshape((x.shape[0], -1)) @ w + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = (dout @ w.T).reshape(x.shape)
  dw = x.reshape(x.shape[0], -1).T @ dout
  db = np.sum(dout, axis=0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[x <= 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
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
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  
  stride, pad = conv_param["stride"], conv_param["pad"]
  padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant')
  N, _, H, _ = x.shape
  F, _, HH, WW = w.shape
  h_out = int(1 + (H + 2 * pad - HH) / stride)
  w_out = int(1 + (H + 2 * pad - WW) / stride)
  out = np.zeros((N, F, h_out, w_out))

  for image_idx in range(N):
    for kernel_idx in range(F):
      for W_idx in range(w_out):
        for H_idx in range(h_out):
          start_W = W_idx * stride
          end_W = W_idx * stride + WW
          start_H = H_idx * stride
          end_H = H_idx * stride + HH

          applied_kernel = padded[image_idx, :, start_H:end_H, start_W:end_W] * w[kernel_idx, :]
          out[image_idx, kernel_idx, H_idx, W_idx] = np.sum(applied_kernel) + b[kernel_idx]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
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
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  
  x, w, b, conv_param = cache
  stride, pad = conv_param["stride"], conv_param["pad"]
  padded = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant')
  N, _, H, W = x.shape
  F, _, HH, WW = w.shape
  h_out = int(1 + (H + 2 * pad - HH) / stride)
  w_out = int(1 + (H + 2 * pad - WW) / stride)

  dx = np.zeros_like(padded)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  for kernel_idx in range(F):
    db[kernel_idx] += np.sum(dout[:, kernel_idx, :, :])

  for image_idx in range(N):
    for kernel_idx in range(F):
      for W_idx in range(w_out):
        for H_idx in range(h_out):
          start_W = W_idx * stride
          end_W = W_idx * stride + WW
          start_H = H_idx * stride
          end_H = H_idx * stride + HH

          dx[image_idx, :, start_H:end_H, start_W:end_W] += dout[image_idx, kernel_idx, H_idx,W_idx] * w[kernel_idx, :]
          dw[kernel_idx, :] += dout[image_idx, kernel_idx, H_idx,W_idx] * padded[image_idx, :, start_H:end_H, start_W:end_W]
                  
  dx = dx[:, :, pad:H+pad, pad:W+pad]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################

  pool_height, pool_width, stride = pool_param.get('pool_height'), pool_param.get('pool_width'), pool_param.get('stride')
  N, C, H, W = x.shape

  h_out = int(((H - pool_height) / stride) + 1)
  w_out = int(((W - pool_width) / stride) + 1)

  out = np.zeros((N, C, h_out, w_out))
  for image_idx in range(N):
    for channel_idx in range(C):
      for W_idx in range(w_out):
        for H_idx in range(h_out):
          start_W = W_idx * stride
          end_W = W_idx * stride + pool_width
          start_H = H_idx * stride
          end_H = H_idx * stride + pool_height
          out[image_idx, channel_idx, H_idx, W_idx] = np.max(x[image_idx, channel_idx, start_H:end_H, start_W:end_W])
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param.get('pool_height'), pool_param.get('pool_width'), pool_param.get('stride')
  N, C, _, _ = x.shape
  _, _, h_out, w_out = dout.shape
  dx = np.zeros_like(x)
  for image_idx in range(N):
    for channel_idx in range(C):
      for W_idx in range(w_out):
        for H_idx in range(h_out):
          start_W = W_idx * stride
          end_W = W_idx * stride + pool_width
          start_H = H_idx * stride
          end_H = H_idx * stride + pool_height

          max_idx = np.argmax(x[image_idx, channel_idx, start_H:end_H, start_W:end_W])
          max_loc = np.unravel_index(max_idx, [pool_height,pool_width])
          dx[image_idx, channel_idx, start_H:end_H, start_W:end_W][max_loc] = dout[image_idx, channel_idx, H_idx, W_idx]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

