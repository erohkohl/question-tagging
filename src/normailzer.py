import numpy as np


def standard(train, test):
    a_train = np.array(train)
    a_test = np.array(test)
    norm_train = (a_train - a_train.mean(axis=0)) / a_train.std(axis=0)
    norm_test = (a_test - a_train.mean(axis=0)) / a_train.std(axis=0)
    return norm_train, norm_test
