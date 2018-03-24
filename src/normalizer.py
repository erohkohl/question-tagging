def standard(train, test):
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
