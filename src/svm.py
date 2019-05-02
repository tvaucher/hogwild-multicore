import math
import os
import random
from array import array
from collections import defaultdict
from ctypes import c_float
from functools import reduce
from multiprocessing import Queue
from multiprocessing.sharedctypes import Array
from operator import add
from time import time

from utils import add_dict, dot, sign


class SVM:
    def __init__(self, learning_rate, lambda_reg, batch_size, dim, lock=False):
        self.__learning_rate = learning_rate
        self.__lambda_reg = lambda_reg
        self.__batch_size = batch_size
        self.__dim = dim
        self.__persistence = 30
        self.__lock = lock
        self.__w = Array(c_float, dim, lock=lock)
        self.__log = Queue()

    def __weights(self):
        w = array('f', self.__w) if self.__lock else self.__w
        return w

    def fit(self, train_set, val_set, max_iter, verbose=False):
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
            mini_batch = random.sample(
                range(len(train_set)), self.__batch_size)
            grad, train_loss = self.step(train_set, mini_batch, w)
            for idx, value in grad.items():  # update step
                self.__w[idx] += self.__learning_rate * value
            w = self.__weights()

            # Logging
            validation_loss = self.loss(val_set, w=w)
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

    def step(self, data, minibatch, w):
        ''' Calculates the gradient and train loss. Add regularizer to the train loss '''
        gradient, train_loss = reduce(lambda x, y: (add_dict(x[0], y[0]), x[1] + y[1]),
                                      map(lambda x: self.calculate_grad_loss(x[0], x[1], w),
                                          (data[i] for i in minibatch)),
                                      (defaultdict(float), 0))
        train_loss /= len(minibatch)

        return gradient, train_loss

    def calculate_grad_loss(self, x, label, w):
        ''' Helper for step, computes the hinge loss and gradient of a single point '''
        xw = dot(x, w)
        if self.misclassification(xw, label):
            delta_w = self.gradient(x, label, w)
        else:
            delta_w = self.l2_reg_grad(w, x)
        return delta_w, self.loss_point(x, label, xw=xw, w=w)

    def loss_point(self, x, label, xw=None, w=None):
        ''' Computes the loss of a single point'''
        if xw is None:
            xw = dot(x, w)
        return self.hinge(xw, label) + self.l2_reg(w, x)

    def loss(self, data, w=None):
        ''' Computes the avg loss (incl regulizer) for a data set '''
        if w is None:
            w = self.__weights()
        return reduce(add, map(lambda x: self.loss_point(x[0], x[1], w=w), data))/len(data)

    def l2_reg(self, w, x):
        ''' Returns the regularization term '''
        return self.__lambda_reg * sum([w[i] ** 2 for i in x.keys()]) / len(x)

    def l2_reg_grad(self, w, x):
        '''Returns the gradient of the regularization term  '''
        return {k: -2 * self.__lambda_reg * w[k] / len(x) for k in x.keys()}

    def hinge(self, xw, label):
        ''' Compute the hinge loss for a dot product and a label '''
        return max(1 - label * xw, 0)

    def hinge_grad(self, x, label):
        ''' Compute minus the gradient of hinge loss (only apply when x dot w * label < 1)'''
        return {k: (v * label) for k, v in x.items()}

    def gradient(self, x, label, w):
        hinge_grad = self.hinge_grad(x, label)
        l2_reg_grad = self.l2_reg_grad(w, x)
        return {k: hinge + l2 for (k, hinge), l2 in
                zip(hinge_grad.items(), l2_reg_grad.values())}

    def misclassification(self, x_dot_w, label):
        ''' Returns true if, for a given point, its hingeloss would be > 0. '''
        return x_dot_w * label < 1

    def accuracy(self, data, w=None):
        ''' Compute the accuracy of the model over a data set of (samples, label) '''
        if w is None:
            w = self.__weights()
        return sum((sign(dot(x, w)) == label) for x, label in data)/len(data)

    def log(self):
        return self.__log.get()
