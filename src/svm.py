import os
import random
from ctypes import c_double
from multiprocessing import Queue
from multiprocessing.sharedctypes import Array
from time import time

import numpy as np
from scipy.sparse import csr_matrix


class SVM:
    def __init__(self, learning_rate, lambda_reg, batch_size, dim, lock=False):
        self.__learning_rate = learning_rate
        self.__lambda_reg = lambda_reg
        self.__batch_size = batch_size
        self.__dim = dim
        self.__persistence = 30
        self.__lock = lock
        self.__w = Array(c_double, dim, lock=lock)
        self.__log = Queue()

    def __weights(self):
        w = self.__w._obj if self.__lock else self.__w
        return np.frombuffer(w)

    def fit(self, train_samples, train_labels, val_samples, val_labels, max_iter, verbose=False):
        ''' Fit the model over the data using train - validation separation over at max_iter iteration'''
        # Housekeeping and initialization
        early_stopping_window = []
        window_smallest = np.inf
        log = []
        w = self.__weights()
        start_time = time()
        pid = os.getpid()
        # SGD
        for i in range(max_iter):
            # Update step
            mini_batch = random.sample(range(train_labels.shape[0]), self.__batch_size)
            grad, train_loss = self.step(train_samples[mini_batch], train_labels[mini_batch], w)
            for idx in grad.nonzero()[1]:  # update step
                self.__w[idx] += self.__learning_rate * grad[0, idx]
            w = self.__weights()

            # Logging
            validation_loss = self.loss(val_samples, val_labels, w=w)
            if verbose: 
                print(f'worker {pid}, iter {i}, train loss : {train_loss}, val loss : {validation_loss}')
            log_iter = {'iter': i, 'time': time() - start_time, 'avg_train_loss': train_loss,
                        'validation_loss': validation_loss}
            log.append(log_iter)

            # Early Stopping criterion
            if(len(early_stopping_window) == self.__persistence):
                early_stopping_window = early_stopping_window[1:]
                early_stopping_window.append(validation_loss)
                if(min(early_stopping_window) > window_smallest):
                    if verbose:
                        print(f'woker {pid} has stopped early {i}')
                    log.append({'early_stop': True})
                    break
                window_smallest = min(early_stopping_window)
            else:
                early_stopping_window.append(validation_loss)

        self.__log.put((pid, log))

    def step(self, data, labels, w):
        ''' Calculates the gradient and train loss. Add regularizer to the train loss '''
        xw = data.dot(w)
        mask = self.misclassification(xw, labels)
        hinge_grad = self.grad_hinge(data, labels, mask)
        l2_grad = self.l2_reg_grad(w, data)
        grad = (hinge_grad - l2_grad).sum(0)

        train_loss = self.loss(data, labels, xw=xw, w=w, mask=mask)

        return grad, train_loss

    def loss(self, data, label, xw=None, w=None, mask=None):
        ''' Computes the avg loss (incl regulizer) for a data set'''
        if w is None:
            w = self.__weights()
        if xw is None:
            xw = data.dot(w)
        if mask is None:
            mask = self.misclassification(xw, label)
        hinge_loss = ((1 - xw * label) * mask).sum()
        return (hinge_loss + self.l2_reg(w, data))/data.shape[0]

    def l2_reg(self, w, data):
        ''' Returns the regularization term '''
        mask = csr_matrix((np.ones(data.nnz), data.indices,
                           data.indptr), shape=data.shape)
        l2 = mask.multiply(w).power(2).sum(1)/mask.getnnz(1).reshape(-1, 1)
        return self.__lambda_reg * l2.sum()

    def l2_reg_grad(self, w, data):
        '''Returns the gradient of the regularization term  '''
        mask = csr_matrix((np.ones(data.nnz), data.indices,
                           data.indptr), shape=data.shape)
        return 2 * self.__lambda_reg * mask.multiply(w).multiply(1/mask.getnnz(1).reshape(-1, 1))

    def grad_hinge(self, data, labels, mask):
        ''' Compute minus the gradient of hinge loss (only apply when x dot w * label < 1 (mask))'''
        return data.multiply(labels.reshape(-1, 1)).multiply(mask.reshape(-1, 1))

    def misclassification(self, x_dot_w, label):
        ''' Returns true if, for a given point, its hingeloss would be > 0. '''
        return x_dot_w * label < 1

    def predict(self, data, w):
        ''' Predict the label of the input data '''
        return np.sign(data.dot(w))

    def accuracy(self, data, labels, w=None):
        ''' Compute the accuracy of the model over a data set of (samples, label) '''
        if w is None:
            w = self.__weights()
        return (self.predict(data, w) == labels).sum()/data.shape[0]

    def log(self):
        return self.__log.get()
