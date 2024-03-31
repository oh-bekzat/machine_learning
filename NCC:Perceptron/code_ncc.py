# please fill all missing lines
# you need to start with the function 'digits'
# then continue with ncc_train and then predict_ncc
# if it the code is completed correctly,
# it will print a pdf in the local folder with results
# happy coding..

import pylab as pl
import scipy as sp
import numpy as np
from scipy.io import loadmat
import pdb


def load_data(fname):
    # load the data
    data = loadmat(fname)
    # extract images and labels
    imgs = data['data_patterns']
    labels = data['data_labels']
    return imgs, labels

def ncc_train(X,Y,Xtest,Ytest):
    # initialize accuracy vector
    acc = sp.zeros(X.shape[-1])
    # unique class labels
    cids = sp.unique(Y)
    # initialize mu, shape should be (256,2) - why? 
    mu = 
    # initialize counter , shape should be (2,) - why?
    Nk = 
    # loop over all data points in training set
    for n 
        # set idx to current class label
        idx = cids==Y[n]
        # update mu
        mu[:,idx] = 
        # update counter
        Nk[idx]
        # predict test labels with current mu
        yhat = predict_ncc(Xtest,mu)
        # calculate current accuracy with test labels
        acc[n] =
    # return weight vector and error
    return mu,acc

def predict_ncc(X,mu):
    # do nearest-centroid classification
    # initialize distance matrix with zeros and shape (602,2) - why?
    NCdist =
    # compute euclidean distance to centroids
    # loop over both classes
    for ic in sp.arange(mu.shape[-1]):
        # calculate distances of every point to centroid
        # in one line
        NCdist[:,ic] = 

    # assign the class label of the nearest (euclidean distance) centroid
    Yclass = NCdist.argmin(axis=1)
    return Yclass

def digits(digit):
    fname = "usps.mat"
    imgs,labels = load_data(fname)
    # we only want to classify one digit 
    labels = sp.sign((labels[digit,:]>0)-.5)

    # please think about what the next lines do
    permidx = sp.random.permutation(sp.arange(imgs.shape[-1]))
    trainpercent = 70.
    stopat = sp.floor(labels.shape[-1]*trainpercent/100.)
    stopat= int(stopat)

    # cut segment data into train and test set into two non-overlapping sets:
    X = 
    Y = 
    Xtest = 
    Ytest =
    #check that shapes of X and Y make sense..
    # it might makes sense to print them
    
    mu,acc_ncc = ncc_train(X,Y,Xtest,Ytest)

    fig = pl.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(acc_ncc*100.)
    pl.xlabel('Iterations')
    pl.title('NCC')
    pl.ylabel('Accuracy [%]')

    # and imshow the weight vector
    ax2 = fig.add_subplot(1,2,2)
    # reshape weight vector
    weights = sp.reshape(mu[:,-1],(int(sp.sqrt(imgs.shape[0])),int(sp.sqrt(imgs.shape[0]))))
    # plot the weight image
    imgh = ax2.imshow(weights)
    # with colorbar
    pl.colorbar(imgh)
    ax2.set_title('NCC Centroid')
    # remove axis ticks
    pl.xticks(())
    pl.yticks(())
    # remove axis ticks
    pl.xticks(())
    pl.yticks(())

    # write the picture to pdf
    fname = 'NCC_digits-%d.pdf'%digit
    pl.savefig(fname)


digits(0)




