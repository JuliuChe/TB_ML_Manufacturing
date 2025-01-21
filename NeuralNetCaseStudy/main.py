# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
#np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import os
import pickle

def addBiasRow(X):
   return np.insert(X, X.shape[1], np.ones(X.shape[0]), axis=1)

def L(X, Y, W, lbda=1.0, delta=1.0):
    """
    Loss Functiion L = 1/N * sum(Li) -> For Support Vector Machine (vs. SoftMax classifier)
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - Y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  - lbda is a lambda, i.e. regularization strength
    """
    scores = W.dot(X)  # 10 x 50,000
    corectClassScore = scores[Y, np.arange(scores.shape[1])] # 1 x 50,000 with correct class score
    margins = np.maximum(0, scores - corectClassScore + delta) # 10 x 50,000
    margins[Y, np.arange(scores.shape[1])] = 0
    # initial understanfing of individual loss_i elements : loss_i = np.sum(margins, axis=0) # 1 x 50,000
    loss_i = np.max(margins, axis=0) # 10 x 50,000
    data_loss = np.sum(loss_i) / X.shape[1] # scalar
    reg_loss= lbda * np.sum(W[:,0:W.shape[1]-1] * W[:,0:W.shape[1]-1])
    loss = data_loss+reg_loss  # add regularization term
    return loss_i, data_loss, reg_loss, loss

def SoftMax(X, Y, W, lbda=1.0):
    """
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - Y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  - lbda is a lambda, i.e. regularization strength

  Li=−log(ef_yi/∑_j(ef_j))

  => cross-entropy between predicted class probabilities q and the correct class p
  H(p,q)=-sum_x(p(x)*log(q(x))) where q=efyi/Sum_j(efyj)

  Here we calculate the cross-entropy loss between the predicted class probabilities and the correct class probabilities
  """

    scores = W.dot(X)  # 10 x 50,000
    norm_scores=scores-np.max(scores, axis=0) # for numerical stability

    loss_i = -norm_scores[Y, np.arange(norm_scores.shape[1])] + np.log(np.sum(np.exp(norm_scores), axis=0)) # 1 x 50,000
    data_loss = np.sum(loss_i) / X.shape[1] # scalar
    reg_loss=lbda * np.sum(W[:,0:W.shape[1]-1] * W[:,0:W.shape[1]-1])
    loss= data_loss+reg_loss  # add regularization term
    return scores, loss_i, data_loss, reg_loss, loss


### CIFAR-10 dataset loading
def load_CIFAR10(cifar10_dir):
    def load_CIFAR_batch(filename):
        # load a single batch of cifar-10 data which contains 10'000 images
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            Y = datadict[b'labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
            return X, Y

    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(cifar10_dir, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_CIFAR_batch(os.path.join(cifar10_dir, 'test_batch'))
    return X_train, y_train, X_test, y_test


def get_CIFAR10_Data(num_training=50000, num_validation=0, num_test=10000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

### END of CIFAR-10 dataset loading ###


### Implementation of a toy Neural Network
### 2 Layer Neural Net
def generate_dataSet(N,D,K):
    X = np.zeros((N*K,D))
    y = np.zeros(N*K, dtype='uint8')
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X,y


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4,5, 6]]) #[[1 2 3],[4 5 6]]
    b=np.r_['-1,2,0', a[0],a[1]]#[[1, 4], [2, 5], [3, 6]]
    c=np.c_[a[0],a[1]]#[[1, 4], [2, 5], [3, 6]]


    print(np.linspace(0*4,(0+1)*4,10))
    print(np.random.randn(10)*0.2)


    X,y = generate_dataSet(100,2,3)

    # lets visualize the data:
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    # plt.show()



    """ Cifar data set loading 
    #Original CIFAR-10 dataset is here : https://www.cs.toronto.edu/~kriz/cifar.html
    
    cifar10_dir = 'datasets/cifar-10-batches-py/data_batch_1'
    d = unpickle(cifar10_dir)
    print(d.keys())
    print('Batch label : %s' %(d[b'batch_label']))
    print('All labels : \n number of rows: %s, number of columns : %s \n albel head: %s \n label tail: %s \n ' %(len(d[b'labels']), 1, d[b'labels'][0], d[b'labels'][-1]))
    print('data :  \n number of rows : %f, number of columns : %f  \n data head (%f): \n   %s  \n data tail (%f): \n   %s' %(d[b'data'].shape[0], d[b'data'].shape[1], len(d[b'data'][0]), d[b'data'][0] , len(d[b'data'][-1]), d[b'data'][-1]))
    print('images filenames : \n number of rows : %s, number of columns : %s  \n image head: \n   %s  \n image tail: \n   %s' %(np.shape(d[b'filenames'])[0], 1, d[b'filenames'][0], d[b'filenames'][-1]))

    X_train, y_train, X_val, y_val, X_test, y_test=get_CIFAR10_Data()
    X_train_rows = X_train.reshape(X_train.shape[0], 32 * 32 * 3)
    X_test_rows = X_test.reshape(X_test.shape[0], 32 * 32 * 3)

    print(X_train_rows.shape)

    X_train_rows_bias = addBiasRow(X_train_rows)
    print(X_train_rows_bias.shape)
    """

    """ Testing of SVM and Softmax classifier 
    Please follow this link to get the testing data : http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/
    
    X_svm = np.array([[0.5,0.8,0.3,-0.4,-0.3,-0.7,0.7,0.5,-0.4],[0.4,0.3,0.8,0.3,0.7,0.2,-0.4,-0.6,-0.5],[1,1,1,1,1,1,1,1,1]])

    y_svm = np.array([0,0,0,1,1,1,2,2,2])

    W = np.array([[1.53,0.97,-0.28], [-1.07, 0.4,0.08], [0.58, -1.90, 0.2]])
    Li_svm, data_loss, reg_loss, loss=L(X_svm, y_svm, W, lbda=0.1, delta=1.0)
    print(Li_svm, data_loss, reg_loss, loss)
    scores, Li_soft, data_loss, reg_loss, loss=SoftMax(X_svm, y_svm, W, lbda=0.10)
    print(scores, Li_soft, data_loss, reg_loss, loss)
    """


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
