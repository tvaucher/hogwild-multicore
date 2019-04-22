import math
import os
import random
from ctypes import c_double
from functools import reduce
from multiprocessing import Process, Queue, RLock, cpu_count
from multiprocessing.sharedctypes import Array
from operator import add
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

    def fit(self, data, validation, max_iter):
        ''' Fit the model over the data using train - validation separation over at max_iter iteration'''
        # Housekeeping and initialization
        early_stopping_window = []
        window_smallest = math.inf
        log = []
        w = self.__weights()
        start_time = time()
        pid = os.getpid()

        # SGD
        for i in range(max_iter):
            # Update step
            subset_indices = random.sample(range(len(data)), self.__batch_size)
            grad, train_loss = self.step(data[subset_indices], w)
            for idx, grad_val in zip(grad.indices, grad.data):
                self.__w[idx] += self.__learning_rate * grad_val
            w = self.__weights()

            # Logging
            validation_loss = self.loss(validation, w=w)
            # print(f'worker {pid}, iter {i}, train loss : {train_loss}, val loss : {validation_loss}')
            log_iter = {'iter': i, 'time': time() - start_time, 'avg_train_loss': train_loss,
                        'validation_loss': validation_loss}  # , 'validation_accuracy': validation_accuracy}
            # print(log_iter)
            log.append(log_iter)

            # Early Stopping criterion
            if(len(early_stopping_window) == self.__persistence):
                early_stopping_window = early_stopping_window[1:]
                early_stopping_window.append(validation_loss)
                if(min(early_stopping_window) > window_smallest):
                    print(f'woker {pid} has stopped early {i}')
                    log.append({'early_stop': True})
                    break
                window_smallest = min(early_stopping_window)
            else:
                early_stopping_window.append(validation_loss)

        self.__log.put((pid, log), block=False)

    def step(self, data, w):
        ''' Calculates the gradient and train loss. Add regularizer to the train loss '''
        gradient, train_loss = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]),
                                      map(lambda x: self.calculate_grad_loss(x[0], x[1], w),
                                          data))
        train_loss /= len(data)

        return gradient, train_loss

    def calculate_grad_loss(self, x, label, w):
        ''' Helper for step, computes the hinge loss and gradient of a single point '''
        xw = x.dot(w)[0]
        if self.misclassification(xw, label):
            delta_w = self.gradient(x, label, w)
        else:
            delta_w = self.reg_gradient(w, x)
        return delta_w, self.loss_point(x, label, xw=xw, w=w)

    def loss_point(self, x, label, xw=None, w=None):
        ''' Computes the loss of a single point'''
        if xw is None:
            xw = x.dot(w)[0]
        return max(1 - label * xw, 0) + self.l2_reg(w, x)

    def loss(self, data, w=None):
        ''' Computes the avg loss (incl regulizer) for a data set '''
        if w is None:
            w = self.__weights()
        return reduce(add, map(lambda x: self.loss_point(x[0], x[1], w=w), data))/len(data)

    def l2_reg(self, w, x):
        ''' Returns the regularization term '''
        return self.__lambda_reg * (w[x.indices] ** 2).sum()/x.nnz

    def l2_reg_grad(self, w, x):
        '''Returns the gradient of the regularization term  '''
        return 2 * self.__lambda_reg * w[x.indices]/x.nnz

    def gradient(self, x, label, w):
        ''' Returns the gradient of the loss with respect to the weights '''
        grad = x * label
        grad.data -= self.l2_reg_grad(w, x)
        return grad

    def reg_gradient(self, w, x):
        ''' Sparse matrice loss, gradient of regularizer '''
        return csr_matrix((-self.l2_reg_grad(w, x), x.indices, x.indptr), (1, self.__dim))

    def misclassification(self, x_dot_w, label):
        ''' Returns true if, for a given point, its hingeloss would be > 0. '''
        return x_dot_w * label < 1

    def predict(self, sample, w):
        ''' Predict the label of the input data '''
        def sign(x): return 1 if x > 0 else -1 if x < 0 else 0
        return sign(sample.dot(w))

    def accuracy(self, data, w=None):
        ''' Compute the accuracy of the model over a data set of (samples, label) '''
        if w is None:
            w = self.__weights()
        return reduce(add, map(lambda x: self.predict(x[0], w) == x[1], data))/len(data)

    def log(self):
        return self.__log.get()
