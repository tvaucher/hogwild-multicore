import argparse
import csv
import math
import json
from datetime import datetime
from multiprocessing import Process, cpu_count
from os import path
from time import time

import settings as s
from load_data import DataLoader
from svm import SVM

def linspace(size, nb_process):
    ''' generator to create a linear space from 0 to `size` of len `nb_process` '''
    start = 0
    step = math.ceil(size/nb_process)
    while start < size:
        yield start
        start += step
    yield size

def get_splits(size, nb_process):
    '''Generate the splits for an array of `size` over `nb_process`'''
    splits = list(linspace(size, nb_process))
    return zip(splits, splits[1:])


def fit_then_dump(model, data, nb_epoch, nb_process, notest=False, verbose=False):
    # Create and launch the processes
    print(f'Starting {nb_process} sub processes')
    processes = [Process(target=model.fit, args=(
        data.training_set[start:end], data.validation_set, nb_epoch, verbose))
        for start, end in get_splits(len(data.training_set), nb_process)]
    start_time = time()
    for p in processes:
        p.start()

    # Get the logs from the process queue
    process_log = []
    for i in range(len(processes)):
        worker_id, l = model.log()
        process_log.append({'id': i, 'worker_id': worker_id, 'log': l})

    # Join the processes
    for p in processes:
        p.join()
    end_time = time()
    print('Done with the sub processes')

    # Compute end results
    training_accuracy = model.accuracy(data.training_set)
    validation_accuracy = model.accuracy(data.validation_set)
    valdiation_loss = model.loss(data.validation_set)
    test_accuracy = model.accuracy(data.test_set) if not notest else 0
    
    # Print results to user
    print(f'Nb epoch per worker (max) : {nb_epoch}, Elapsed time : {end_time - start_time:.1f} sec')
    print(f'Train Accuracy : {training_accuracy:.4f}%, Validation Accuracy : {validation_accuracy:.4f}% , Test Accuracy : {test_accuracy:.4f}%')
    # Save results in a log
    log = [{'start_time': datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            'running_time': end_time - start_time,
            'training_accuracy': training_accuracy,
            'validation_accuracy': validation_accuracy,
            'validation_loss': valdiation_loss,
            'test_accuracy': test_accuracy,
            'nb_epoch': nb_epoch,
            'fit_log': process_log,
            }]

    logname = f'{datetime.utcfromtimestamp(end_time).strftime("%Y%m%d_%H%M%S")}_log.json'
    with open(path.join(s.logpath, logname), 'w') as outfile:
        json.dump(log, outfile)

    return training_accuracy, validation_accuracy, valdiation_loss, test_accuracy


def grid_search(data, learning_rates, lambdas, batch_sizes):
    values = [(
        'learning_rate',
        'lambda_reg',
        'batch_size',
        'training_accuracy',
        'validation_accuracy',
        'validation_loss',
        'test_accuracy'
    )]
    for learning_rate in learning_rates:
        for lambda_reg in lambdas:
            for bsize in batch_sizes:
                model = SVM(learning_rate, lambda_reg, bsize, s.dim)
                training_accuracy, validation_accuracy, valdiation_loss, test_accuracy = fit_then_dump(
                    model, data, 100, cpu_count())
                values.append((learning_rate, lambda_reg, bsize,
                               training_accuracy, validation_accuracy, valdiation_loss, test_accuracy))

    with open(path.join(s.logpath, datetime.utcfromtimestamp(time()).strftime("%Y%m%d_%H%M%S") + '_grid_search_results.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l', '--lock', help='Use lock instead of Lock-free Hogwild', action='store_true')
    parser.add_argument(
        '-n', '--niter', help='Number of iteration, default=400', type=int, default=400)
    parser.add_argument('-p', '--process', help='Number of processes to run, default=cpu_count',
                        type=int, default=int(cpu_count()))
    parser.add_argument(
        '--gridsearch', help='Perfom a grid search over the hyperparameters', action='store_true')
    parser.add_argument('--notest', help='Doesn\'t load and compute the test set / accuracy', action='store_true')
    parser.add_argument('-v', '--verbose', help='Print the intermediate train / val loss at each iteration', action='store_true')
    args = parser.parse_args()

    data = DataLoader()
    # workers = [1, 2, 4, 8]
    # bsizes = [1, 10, 100, 500, 1000]
    # for w in workers:
    #     model = SVM(s.learning_rate, s.lambda_reg,
    #                 s.batch_size, s.dim, lock=args.lock)
    #     fit_then_dump(model, data, args.niter, w)
    #     data.shuffle_train_val_data()

    # for bsize in bsizes:
    #     lr = s.learning_rate * 100 / bsize
    #     model = SVM(lr, s.lambda_reg,
    #                 s.batch_size, s.dim, lock=args.lock)
    #     fit_then_dump(model, data, args.niter, args.process)
    #     data.shuffle_train_val_data()

    if not args.gridsearch:
        lr = s.learning_rate / args.process if args.lock else s.learning_rate
        model = SVM(lr, s.lambda_reg,
                    s.batch_size, s.dim, lock=args.lock)
        fit_then_dump(model, data, args.niter, args.process, notest=args.notest, verbose=args.verbose)
    else:
        learning_rates = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.08]
        batch_sizes = [100]
        lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        for i in range(3):
            grid_search(data, learning_rates, lambdas, batch_sizes)
            data.shuffle_train_val_data()
