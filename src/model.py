import logging

import matplotlib.pyplot as plt
import tensorflow as tf

import csv_helper
import encoder
import normalizer as norm

N_INPUT = 800
N_TRAIN = int(N_INPUT * 0.9)
N_TEST = int(N_INPUT * 0.1)
N_CLASSES = 3
LEARNING_RATE = 0.001
N_EPOCHS = 1000000
N_STEPS = 100
N_HIDDEN_NEURONS = 128
N_HIDDEN_LAYERS = 1

# Anonymous functions for adding sigmoid and softmax layer as wells as
# for initializing variables with zeros and uniform random values between
# -1 and +1.
act = lambda l, w, b: tf.nn.relu(tf.add(tf.matmul(l, w), b))
soft = lambda l, w, b: tf.nn.softmax(tf.add(tf.matmul(l, w), b))
zeros = lambda h: tf.Variable(tf.zeros([h]))
random = lambda i, o: tf.Variable(tf.random_uniform([i, o], -1, 1))


def accuracy(prediction, target) -> float:
    n_correct = 0
    n = len(prediction)
    for p, t in zip(prediction, target):
        # get most likely class
        for i in range(0, len(p)):
            if p[i] == max(p) and t[i] == 1:
                n_correct += 1
    return (float(n_correct) / float(n)) * 100.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)
    model_file = "data/trained_models/model-" + str(N_HIDDEN_LAYERS) + "-" + str(N_HIDDEN_NEURONS) + ".ckpt"

    # Read data from dump and map each label to it's one hot vector
    csv_input, csv_output = csv_helper._import('data/tagged_questions.csv', N_CLASSES)
    for i in range(0, len(csv_output)):
        csv_output[i] = encoder.one_hot(int(csv_output[i]), N_CLASSES)
    input_size = len(csv_input[0])

    # Split data into training and test set
    train_input, train_output = csv_input[:N_TRAIN], csv_output[:N_TRAIN]
    test_input, test_output = csv_input[N_TRAIN:N_TRAIN + N_TEST], csv_output[N_TRAIN:N_TRAIN + N_TEST]
    train_input, test_input = norm.standard(train_input, test_input)

    print(train_input)
    print(train_output)

    # Set TensorFlows placeholder for training input, target value and test input
    ph_input = tf.placeholder(tf.float32, shape=[N_TRAIN, input_size])
    ph_target = tf.placeholder(tf.float32, shape=[N_TRAIN, N_CLASSES])
    ph_test = tf.placeholder(tf.float32, shape=[N_TEST, input_size])

    # Setup neural net: weights, biases and connect layers
    weights = [random(input_size, N_HIDDEN_NEURONS)]
    biases = [zeros(N_HIDDEN_NEURONS)]
    layers = [act(ph_input, weights[0], biases[0])]
    test_layers = [act(ph_test, weights[0], biases[0])]

    for i in range(1, N_HIDDEN_LAYERS + 1):
        print(i)
        print(weights)
        weights.append(random(N_HIDDEN_NEURONS, N_HIDDEN_NEURONS))
        biases.append(zeros(N_HIDDEN_NEURONS))
        layers.append(act(layers[i - 1], weights[i], biases[i]))
        test_layers.append(act(test_layers[i - 1], weights[i], biases[i]))

    weights.append(random(N_HIDDEN_NEURONS, N_CLASSES))
    biases.append(zeros(N_CLASSES))
    layers.append(soft(layers[N_HIDDEN_LAYERS], weights[N_HIDDEN_LAYERS + 1], biases[N_HIDDEN_LAYERS + 1]))
    test_layers.append(
        soft(test_layers[N_HIDDEN_LAYERS], weights[N_HIDDEN_LAYERS + 1], biases[N_HIDDEN_LAYERS + 1]))

    # Add node to graph that calculates mean squared error
    LOSS = tf.nn.l2_loss(layers[N_HIDDEN_LAYERS + 1] - ph_target)

    # Initialize optimizer that uses gradient descent
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(LOSS)

    saver = tf.train.Saver()
    sess = tf.Session()

    try:
        saver.restore(sess, model_file)
    except:
        init = tf.global_variables_initializer()
        sess.run(init)

    losses = []
    epoch = 0

    # Run training and check depended on N_STEPS the relative number of correct
    # classified training and test samples as well as plot the loss function in
    # addition to it's number of epochs.
    acc_train = accuracy(sess.run(layers[N_HIDDEN_LAYERS + 1], feed_dict={ph_input: train_input}), train_output)
    while acc_train < 90.0:
        train_dict = {ph_input: train_input, ph_target: train_output}
        sess.run(optimizer, feed_dict=train_dict)
        i += 1
        if i % N_STEPS == 0:
            acc_train = accuracy(sess.run(layers[N_HIDDEN_LAYERS + 1], feed_dict={ph_input: train_input}), train_output)
            acc_test = accuracy(sess.run(test_layers[N_HIDDEN_LAYERS + 1], feed_dict={ph_test: test_input}),
                                test_output)
            actual_loss = sess.run(LOSS, feed_dict=train_dict)
            losses.append(actual_loss)

            plt.plot(losses, 'b')
            plt.xlabel('Iterations *1000 ')
            plt.ylabel('Loss ')
            plt.draw()
            plt.pause(0.001)
            logger.info('-     Epoch = ' + str(i) + ', Loss = ' + str(actual_loss) + ', Train Coverage: ' +
                        str(acc_train) + '%' + ', Test Coverage: ' + str(acc_test) + '%')

    saver.save(sess, model_file)
    plt.savefig('data/loss.png')
