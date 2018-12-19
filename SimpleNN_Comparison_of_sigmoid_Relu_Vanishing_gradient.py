
#For exploding gradient problem:
 #First introduced by mikolov. Its states that if value is above threshold value just cap it to 5 or something low.
 #Although being simplle it makes huge difference in RNNs
 #So why clipping doesnt solve vanishing gradient problem? because in vanishing gradient value gets smaller and smaller
 #and it doesnt make sense to clip it.. so why not just bump it?, consider this scenerio where if we bump it by some value then
 #its like trying to say hey 50th word doesnt make sense lets go to 100th word which will (?) make sense.. so it doesnt work here either..
#For this, we will initialize Ws to identity matrix I and f(z) = rect(z) = max(z,0) or try LSTMs..
import numpy as np
import matplotlib.pyplot as plt
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()
# initialize parameters randomly
W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))
# compute class scores for a linear classifier
scores = np.dot(X, W) + b
num_examples = X.shape[0]
# get unnormalized probabilities
exp_scores = np.exp(scores)
# normalize them for each example
probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
correct_logprobs = -np.log(probs[range(num_examples),y])
# compute the loss: average cross-entropy loss and regularization
data_loss = np.sum(correct_logprobs)/num_examples
reg = 1e-3
# some hyperparameters
step_size = 1e-0
reg_loss = 0.5*reg*np.sum(W*W)
loss = data_loss + reg_loss
dscores = probs
dscores[range(num_examples),y] -= 1
dscores /= num_examples
dW = np.dot(X.T, dscores)
db = np.sum(dscores, axis=0, keepdims=True)
dW += reg*W # don't forget the regularization gradient
W += -step_size * dW
b += -step_size * d
