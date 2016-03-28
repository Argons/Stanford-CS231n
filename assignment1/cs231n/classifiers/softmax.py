import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
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
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_sample = X.shape[0]
  for i in xrange(num_sample):
    f = X[i].dot(W)
    f -= f.max()
    prob = np.exp(f)
    prob /= prob.sum()
    loss -= np.log(prob[y[i]])
    dW += X[i].reshape(-1,1) * prob
    dW[:,y[i]] -= X[i]
  loss /= num_sample
  loss += 0.5 * reg * np.sum(W*W)
  dW /= num_sample
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = X.dot(W)
  f -= f.max(axis=1).reshape(-1, 1)
  f = np.exp(f)
  f /= f.sum(axis=1).reshape(-1, 1)
  loss = np.log(f[xrange(y.shape[0]),y])
  loss = -loss.sum()
  loss /= y.shape[0]
  loss += 0.5 * reg * np.sum(W*W)

  dW = X.T.dot(f)
  dW -= X.T.dot(y.reshape(-1,1)==xrange(dW.shape[1]))
#  for i in xrange(dW.shape[1]):
#      dW[:,i] -= X[y==i].sum(axis=0)
  dW /= y.shape[0]
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
