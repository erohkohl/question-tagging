import tensorflow as tf

N_TRAIN = 100
N_CLASSES = 20
LEARNING_RATE = 0.000001
N_EPOCHS = 10000000
N_HIDDEN_NEURONS = 256

if __name__ == "__main__":
    train_input, train_output = preprocess_data()
    n_input = len(train_input[0])

    input = tf.placeholder(tf.float32, shape=[N_TRAIN, n_input])
    target = tf.placeholder(tf.float32, shape=[N_TRAIN, N_CLASSES])

    W_1 = tf.Variable(tf.random_uniform([n_input, N_HIDDEN_NEURONS], -1, 1))
    W_2 = tf.Variable(tf.random_uniform([N_HIDDEN_NEURONS, N_HIDDEN_NEURONS], -1, 1))
    W_3 = tf.Variable(tf.random_uniform([N_HIDDEN_NEURONS, N_HIDDEN_NEURONS], -1, 1))
    W_4 = tf.Variable(tf.random_uniform([N_HIDDEN_NEURONS, N_HIDDEN_NEURONS], -1, 1))
    W_5 = tf.Variable(tf.random_uniform([N_HIDDEN_NEURONS, N_HIDDEN_NEURONS], -1, 1))
    W_6 = tf.Variable(tf.random_uniform([N_HIDDEN_NEURONS, N_CLASSES], -1, 1))

    B_1 = tf.Variable(tf.zeros([N_HIDDEN_NEURONS]))
    B_2 = tf.Variable(tf.zeros([N_HIDDEN_NEURONS]))
    B_3 = tf.Variable(tf.zeros([N_HIDDEN_NEURONS]))
    B_4 = tf.Variable(tf.zeros([N_HIDDEN_NEURONS]))
    B_5 = tf.Variable(tf.zeros([N_HIDDEN_NEURONS]))
    B_6 = tf.Variable(tf.zeros([N_CLASSES]))

    hidden_layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input, W_1), B_1))
    hidden_layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, W_2), B_2))
    hidden_layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, W_3), B_3))
    hidden_layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, W_4), B_4))
    hidden_layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer_1, W_5), B_5))
    output_layer = tf.nn.softmax(tf.add(tf.matmul(hidden_layer_2, W_6), B_6))

    LOSS = tf.nn.l2_loss(output_layer - target)
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(LOSS)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    print("graph init DONE")
    for i in range(N_EPOCHS):
        train_dict = feed_dict = {input: train_input[:N_TRAIN], target: train_output[:N_TRAIN]}
        sess.run(optimizer, feed_dict=train_dict)
        if i % 10000 == 0:
            print('Epoch = ', i, ', Loss = ', sess.run(LOSS, feed_dict=train_dict))
