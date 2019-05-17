import os

if os.environ.get('LOCAL', False):
    path = '..'
    logpath = '../logs'
elif os.environ.get('DEBUG', False):
    path = 'D:\\Documents\\EPFL\\git\\hogwild-multicore'
    logpath = 'D:\\Documents\\EPFL\\git\\hogwild-multicore\\logs'
else:
    path = '/data'
    logpath = '/data/logs'
dim = 47238
learning_rate = 0.015
lambda_reg = 1e-4
batch_size = 100

validation_frac = 0.1
