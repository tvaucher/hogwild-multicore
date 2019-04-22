import os

if os.environ.get('LOCAL', False):
    path = '..'
    logpath = '../logs'
elif os.environ.get('DEBUG', False):
    path = 'D:\\Documents\\EPFL\\git\\hogwild-multicore'
    logpath = 'D:\\Documents\\EPFL\\git\\hogwild-multicore'
else:
    path = '/data'
    logpath = '/data/logs'
dim = 47238
learning_rate = 0.03
lambda_reg = 1e-5
batch_size = 100

validation_frac = 0.1
