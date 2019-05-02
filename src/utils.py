def dot(x, w):
    ''' Calculates the dotproduct for sparse x and w. '''
    return sum([v * w[k] for k, v in x.items()])

def sign(x):
    ''' Sign function '''
    return 1 if x > 0 else -1 if x < 0 else 0

def add_dict(acc, x):
    for k, v in x.items():
        acc[k] += v
    return acc