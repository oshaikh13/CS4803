import numpy as np
from random import shuffle

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  N = X.shape[1]

  f = W @ X
  f -= np.max(f, axis=0)
  p = np.exp(f) / np.sum(np.exp(f), axis=0, keepdims=True)

  log_p = np.log(p[y, range(N)])
  loss = -np.sum(log_p) / N + reg * np.sum(W * W)
     
  p[y, range(N)] -= 1
  dW = 1.0/N * (p @ X.T) + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
