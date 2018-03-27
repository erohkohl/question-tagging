import logging

import matplotlib.pyplot as plt
import tensorflow as tf

from util import encoder, csv_helper, normalizer as norm

N_INPUT = 50000
N_TRAIN = int(N_INPUT / 0.9)
N_TEST = int(N_INPUT / 0.9 * 0.1)
N_CLASSES = 50
LEARNING_RATE = 0.001
N_EPOCHS = 20000
N_STEPS = 100
N_HIDDEN_NEURONS = [10, 12, 24, 48]
N_HIDDEN_LAYERS = len(N_HIDDEN_NEURONS)

# Anonymous functions for adding sigmoid and softmax layer as wells as
# for initializing variables with zeros and uniform random values between
# -1 and +1.
act = lambda l, w, b: tf.nn.relu(tf.add(tf.matmul(l, w), b))
soft = lambda l, w, b: tf.nn.softmax(tf.add(tf.matmul(l, w), b))
zeros = lambda h: tf.Variable(tf.zeros([h]))
random = lambda i, o: tf.Variable(tf.random_uniform([i, o], -1, 1))


def build_model():
    # Setup neural net: weights, biases and connect layers
    weights = [random(input_size, N_HIDDEN_NEURONS[0])]
    biases = [zeros(N_HIDDEN_NEURONS[0])]
    layers = [act(ph_in, weights[0], biases[0])]
    test_layers = [act(ph_test_in, weights[0], biases[0])]

    # for i in range(1, N_HIDDEN_LAYERS + 1):
    for i in range(1, len(N_HIDDEN_NEURONS)):
        weights.append(random(N_HIDDEN_NEURONS[i - 1], N_HIDDEN_NEURONS[i]))
        biases.append(zeros(N_HIDDEN_NEURONS[i]))
        layers.append(act(layers[i - 1], weights[i], biases[i]))
        test_layers.append(act(test_layers[i - 1], weights[i], biases[i]))

    weights.append(random(N_HIDDEN_NEURONS[N_HIDDEN_LAYERS - 1], N_CLASSES))
    biases.append(zeros(N_CLASSES))
    layers.append(soft(layers[N_HIDDEN_LAYERS - 1], weights[N_HIDDEN_LAYERS], biases[N_HIDDEN_LAYERS]))
    test_layers.append(
        soft(test_layers[N_HIDDEN_LAYERS - 1], weights[N_HIDDEN_LAYERS], biases[N_HIDDEN_LAYERS]))

    return layers[N_HIDDEN_LAYERS], test_layers[N_HIDDEN_LAYERS]


def record():
    plt.plot(losses, 'b')
    plt.ylabel('Loss ')
    plt.draw()
    plt.pause(0.001)
    logger.info('- Epoch = ' + str(epoch) + ', Loss = ' + str(actual_loss) + ', Train Coverage: ' +
                str(acc_train) + ', Test Coverage: ' + str(acc_test))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)
    model_file = "data/trained_models/model-" + str(N_HIDDEN_LAYERS) + "-" + str(N_HIDDEN_NEURONS) + ".ckpt"

    # Read data from dump and map each label to its one hot vector
    csv_input, csv_output = csv_helper._import('data/tagged_questions.csv', N_CLASSES)
    for i in range(0, len(csv_output)):
        csv_output[i] = encoder.one_hot(int(csv_output[i]), N_CLASSES)
    input_size = len(csv_input[0])

    # Split data into training and test set
    train_in, train_out = csv_input[:N_TRAIN], csv_output[:N_TRAIN]
    test_in, test_out = csv_input[N_TRAIN:N_TRAIN + N_TEST], csv_output[N_TRAIN:N_TRAIN + N_TEST]
    train_in, test_in = norm.standard(train_in, test_in)

    # Set TensorFlows placeholder for training input, target value and test input
    ph_in = tf.placeholder(tf.float32, shape=[N_TRAIN, input_size])
    ph_out = tf.placeholder(tf.float32, shape=[N_TRAIN, N_CLASSES])
    ph_test_in = tf.placeholder(tf.float32, shape=[N_TEST, input_size])
    ph_test_out = tf.placeholder(tf.float32, shape=[N_TEST, N_CLASSES])

    net, test_net = build_model()

    # Add node to graph that calculates mean squared error
    LOSS = tf.nn.l2_loss(net - ph_out)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(ph_out, 1)), tf.float32))

    # Initialize optimizer that uses gradient descent
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(LOSS)

    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    losses = []
    epoch = 0

    # Run training and check depended on N_STEPS the relative number of correct
    # classified training and test samples as well as plot the loss function in
    # addition to its number of epochs.
    acc_train = sess.run(accuracy, feed_dict={ph_in: train_in, ph_out: train_out})
    while epoch < N_EPOCHS:
        train_dict = {ph_in: train_in, ph_out: train_out}
        test_dict = {ph_test_in: test_in, ph_test_out: test_out}
        sess.run(optimizer, feed_dict=train_dict)
        epoch += 1

        if epoch % N_STEPS == 0:
            acc_train = sess.run(accuracy, feed_dict=train_dict)
            acc_test = sess.run(tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(test_net, 1), tf.argmax(ph_test_out, 1)), tf.float32)), feed_dict=test_dict)
            actual_loss = sess.run(LOSS, feed_dict=train_dict)
            losses.append(actual_loss)
            record()

    saver.save(sess, model_file)
    plt.savefig('data/loss.png')
