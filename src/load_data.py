import random
from functools import wraps
from os import path

import numpy as np
from scipy.sparse import csr_matrix

import settings as s


def memoize(f):
    '''[Lazy val in Python](https://stackoverflow.com/questions/18142919/python-equivalent-of-scalas-lazy-val)'''
    @wraps(f)
    def memoized(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))  # make args hashable
        result = memoized._cache.get(key, None)
        if result is None:
            result = f(*args, **kwargs)
            memoized._cache[key] = result
        return result
    memoized._cache = {}
    return memoized


def line_to_features(r, cat):
    '''Transform a line into a feature sparse matrix'''
    r = r.strip().split(' ')
    features = [feature.split(':') for feature in r[2:]]
    col_idx = np.array([0] + [int(idx) + 1 for idx, _ in features])
    row_idx = np.array([0]*(len(features) + 1))
    data = np.array([1.] + [float(value) for _, value in features])
    label = 1 if int(r[0]) in cat else -1
    return (csr_matrix((data, (row_idx, col_idx)), shape=(1, s.dim)), label)


class DataLoader:
    def __init__(self, category='CCAT'):
        print('Reading the doc category and generating training and validation set')
        self.doc_category = self.read_category(category)
        self.training_set, self.validation_set = self.read_train_val_data()

    def read_category(self, category='CCAT'):
        with open(path.join(s.path, 'datasets/rcv1-v2.topics.qrels')) as f:
            content = [l.strip().split(' ') for l in f.readlines()]
            return {int(l[1]) for l in content if l[0] == category}

    def read_train_val_data(self):
        with open(path.join(s.path, 'datasets/lyrl2004_vectors_train.dat')) as f:
            content = f.readlines()
            sets = np.asarray(
                [line_to_features(l, self.doc_category) for l in content])
            indices = np.random.permutation(len(sets))
            separater = int(np.floor(s.validation_frac * len(sets)))
            return sets[indices[separater:]], sets[indices[:separater]]

    @property
    @memoize
    def test_set(self):
        print('Reading the test set')
        content = []
        for i in range(4):
            with open(path.join(s.path, 'datasets/lyrl2004_vectors_test_pt'+str(i)+'.dat')) as f:
                content += f.readlines()
        return np.asarray([line_to_features(l, self.doc_category) for l in content])
