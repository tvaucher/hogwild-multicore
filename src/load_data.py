import math
import random
from functools import wraps
from os import path

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
    '''Transform a line into a feature dict + the according label'''
    r = r.strip().split(' ')
    features = [feature.split(':') for feature in r[2:]]
    feature_dict = {0: 1.}
    for idx, value in features:
        feature_dict[int(idx) + 1] = float(value)
    label = 1 if int(r[0]) in cat else -1
    return (feature_dict, label)


class DataLoader:
    def __init__(self, category='CCAT'):
        print('Reading the doc category and generating training and validation set')
        self.doc_category = self.read_category(category)
        self.shuffle_train_val_data()

    def read_category(self, category='CCAT'):
        with open(path.join(s.path, 'datasets/rcv1-v2.topics.qrels')) as f:
            content = [l.strip().split(' ') for l in f.readlines()]
            return {int(l[1]) for l in content if l[0] == category}

    @memoize
    def read_train_val_data(self):
        with open(path.join(s.path, 'datasets/lyrl2004_vectors_train.dat')) as f:
            content = f.readlines()
            return [line_to_features(l, self.doc_category) for l in content]

    def shuffle_train_val_data(self):
        sets = self.read_train_val_data()
        random.shuffle(sets)
        separater = math.floor(s.validation_frac * len(sets))
        self.training_set = sets[separater:]
        self.validation_set = sets[:separater]

    @property
    @memoize
    def test_set(self):
        print('Reading the test set')
        content = []
        for i in range(4):
            with open(path.join(s.path, 'datasets/lyrl2004_vectors_test_pt'+str(i)+'.dat')) as f:
                content += f.readlines()
        return [line_to_features(l, self.doc_category) for l in content]
