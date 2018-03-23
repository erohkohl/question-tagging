import numpy as np


def standard(train, test):
    #a_train = np.array(train)
    #a_test = np.array(test)
    #norm_train = (a_train - a_train.mean(axis=0)) / a_train.std(axis=0)
    #norm_test = (a_test - a_train.mean(axis=0)) / a_train.std(axis=0)
    norm_train = []
    norm_test = []

    for x in train:
        z = []
        for x_i in x:
            z.append((x_i - min(x)) / (max(x) - min(x)))
        norm_train.append(z)

    for x in test:
        z = []
        for x_i in x:
            z.append((x_i - min(x)) / (max(x) - min(x)))
        norm_test.append(z)

    return norm_train, norm_test
