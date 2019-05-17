# Hogwild-multicore

Welcome to this implementation of the sparse SVM problem described in the original [Hogwild !](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf) paper using the numerous core of a computer. This was implemented during the spring of 2019 in the CS-449 Systems for Data Sciences at EPFL. This is the second milestone of a project including a [synchronous and asynchronous distributed version of Hogwild](https://github.com/liabifano/hogwild-python) implement in Spring 2018 and a [Spark version](https://github.com/tvaucher/hogwild-spark) implemented in the first milestone.

## Prerequisites and Setup

We suppose that you have Anaconda installed on your computer in order to create a conda environment and that you downloaded the datasets on your machine (if you want to use the scipy version of hogwild-multicore, otherwise python 3.7 is enough)

```shell
conda create -n hogwild-multicore python=3.7 numpy scipy
source activate hogwild-multicore
```

## Run

You may need to adapt the paths in `settings.py` to suit your architecture then from the `src` folder :

```shell
python hogwild.py
```

You can look at the help using `python hogwild.py -h`

**Parameters**

- `-l` or `--lock` Use a lock version of Lock-free Hogwild to avoid concurrent update
- `-n` or `--niter` To select the number of iterations of SGD *default : 400*
- `-p` or `--process` To select the number of process to run onto *default : os.cpu_count* 
- `-v`or `--verbose` To show the train and accuracy loss at train time
- `--notest` To run only the training step and not load the test set / compute test accuracy