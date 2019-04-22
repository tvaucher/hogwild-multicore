import argparse
import csv
import json
from datetime import datetime
from multiprocessing import Process, cpu_count
from os import path
from time import time

import numpy as np

import settings as s
from load_data import DataLoader
from svm import SVM


def get_splits(size, nb_process):
    '''Generate the splits for an array of `size` over `nb_process`'''
    splits = np.linspace(0, size, num=(nb_process + 1), dtype=np.uint)
    return zip(splits, splits[1:])


def fit_then_dump(model, data, nb_epoch, nb_process):
    # Create and launch the processes
    print(f'Starting {nb_process} sub processes')
    processes = [Process(target=model.fit, args=(
        data.training_set[start:end], data.validation_set, nb_epoch, ))
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
    test_accuracy = model.accuracy(data.test_set)

    # Print results to user
    print(f'Nb epoch per worker (max) : {nb_epoch}, Elapsed time : {end_time - start_time:.1f} sec')
    print(f'Train Accuracy : {training_accuracy:.4f}%, Validation Accuracy : {validation_accuracy:.4f}%, Test Accuracy : {test_accuracy:.4f}%')
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

    return training_accuracy, validation_accuracy, valdiation_loss

# def grid_search(data, learning_rates, lambdas, batch_fracs):
#     values = [(
#         'learning_rate',
#         'lambda_reg',
#         'frac',
#         'training_accuracy',
#         'validation_accuracy',
#         'validation_loss'
#     )]
#     for learning_rate in learning_rates:
#         for lambda_reg in lambdas:
#             for frac in batch_fracs:
#                 training_accuracy, validation_accuracy, valdiation_loss = fit_then_dump(
#                     data, learning_rate, lambda_reg, frac, niter=100)
#                 values.append((learning_rate, lambda_reg, frac,
#                                training_accuracy, validation_accuracy, valdiation_loss))

#     with open(path.join(s.logpath, datetime.utcfromtimestamp(time()).strftime("%Y%m%d_%H%M%S") + '_grid_search_results.csv'), 'w') as f:
#         writer = csv.writer(f)
#         writer.writerows(values)


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
    args = parser.parse_args()

    data = DataLoader()
    model = SVM(s.learning_rate, s.lambda_reg,
                s.batch_size, s.dim, lock=args.lock)
    fit_then_dump(model, data, args.niter, args.process)

    # learning_rates = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
    # batch_fracs = [0.005, 0.01, 0.02]
    # lambdas = [1e-6, 1e-5, 1e-4, 1e-3]
    # for i in range(4):
    #     grid_search(data, learning_rates, lambdas, batch_fracs)
