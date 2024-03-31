import pylab as pl
import numpy as np
import scipy as sp
from scipy.linalg import eig
from scipy.io import loadmat
import pdb

def load_data(fname):
    # load the data
    data = loadmat(fname)
    # extract images and labels
    X = data['X']
    Y = data['Y']
    # collapse the time-electrode dimensions
    X = sp.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
    # transform the labels to (-1,1)
    Y = sp.sign((Y[0,:]>0) -.5)
    # pick only first 500 (1000, 3000) datapoints and compare optimal shrinkage
    X= ...
    Y= ...
    print(X.shape)
    return X,Y


def train_lda(X,Y,gamma):
    '''
    Train a nearest centroid classifier
    '''
    # class means
    mupos = sp.mean(X[:,Y>0],axis=1)
    muneg = sp.mean(X[:,Y<0],axis=1)

    # inter and intra class covariance matrices
    Sinter = sp.outer(mupos-muneg,mupos-muneg)
    #Sinter = sp.outer(muneg-mupos,muneg-mupos)
    Sintra = sp.cov(X[:,Y>0]) + sp.cov(X[:,Y<0])
    # shrink covariance matrix estimate
    Sintra = ...
    #Sintra = sp.cov(X[:,Y>0]) + sp.cov(X[:,Y<0])
    # solve eigenproblem
    eigvals, eigvecs = sp.linalg.eig(Sinter,Sintra)
    # weight vector
    w = eigvecs[:,eigvals.argmax()]
    # offset
    b = (w.dot(mupos) + w.dot(muneg))/2.
    # return the weight vector
    return w,b


def crossvalidate_nested(X,Y,f,gammas):
    ''' 
    Optimize shrinkage parameter for generalization performance 
    Input:	X	data (dims-by-samples)
                Y	labels (1-by-samples)
                f	number of cross-validation folds
                gammas	a selection of shrinkage parameters
                trainfunction 	trains linear classifier, returns weight vector and bias term
    '''
    # the next two lines reshape vector of indices in to a matrix:
    # number of rows = # of folds
    # number of columns = # of total data-points / # folds
    N = f*int(np.floor(X.shape[-1]/f))
    idx = sp.reshape(sp.arange(N),(f,int(np.floor(N/f)))) 
    pdb.set_trace()
    acc_test = sp.zeros((f))
    testgamma = sp.zeros((gammas.shape[-1],f))
    
    # loop over folds:
    # select one row of 'idx' for testing, all other rows for training
    # call variables (indices) for training and testing 'train' and 'test'
    for ifold in sp.arange(f):
        ...
        ...
        
        # loop over gammas
        for igamma in range(gammas.shape[-1]):
            # each gamma is fed into the inner CV via the function 'crossvalidate_lda'
            # the resulting variable is called 'testgamma'
            testgamma[igamma,ifold] = crossvalidate_lda(...,...,...,...)
        # find the the highest accuracy of gammas for a given fold and use it to train an LDA on the training data
        ...
        w,b = train_lda(...,...,...)
        # calculate the accuracy for this LDA classifier on the test data
        ...
        acc_test[ifold] = ...

    # do some plotting
    pl.figure()
    pl.boxplot(testgamma.T)
    pl.xticks(sp.arange(gammas.shape[-1])+1,gammas)
    pl.xlabel('$\gamma$')
    pl.ylabel('Accuracy')
    pl.savefig('cv_nested-boxplot.pdf')

    return acc_test,testgamma


def crossvalidate_lda(X,Y,f,gamma):
    ''' 
    Test generalization performance of shrinkage lda
    Input:	X	data (dims-by-samples)
                Y	labels (1-by-samples)
                f	number of cross-validation folds
                trainfunction 	trains linear classifier, returns weight vector and bias term
    '''
    N = f*int(np.floor(X.shape[-1]/f))
    idx = sp.reshape(sp.arange(N),(f,int(np.floor(N/f))))
    acc_test = sp.zeros((f))
    
    # loop over folds
    # select one row of idx for testing, all others for training
    # call variables (indices) for training and testing 'train' and 'test'
    for ifold in sp.arange(f):
        ...
        ...
        # train LDA classifier with training data and given gamma:
        w,b = train_lda(...,...,...)
        # test classifier on test data:
        ...
        acc_test[ifold] = ...
    return acc_test.mean()


X,Y = load_data('bcidata.mat')
gammas=sp.array([0,.005,.05,.5,1])
a,b = crossvalidate_nested(X,Y,10,gammas)
print(a)
print(b)

