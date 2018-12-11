import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# Defining input data
x_data = np.arange(100, step=.1, dtype=np.float32)
y_data = x_data + 20 * np.sin(x_data/10)

# Define data size and batch size
n_samples = 1000
batch_size = 100

# Tensorflow is finicky about shapes, so resize
x_train = np.reshape(x_data, (n_samples, 1))
y_train = np.reshape(y_data, (n_samples, 1))

# Define placeholders for input
x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# Define variables to be learned

W = tf.get_variable("weights", shape=(1, 1), initializer=tf.random_normal_initializer())
b = tf.get_variable("bias", (1,), initializer=tf.constant_initializer(0.0))
y_pred = tf.matmul(x, W) + b
loss = tf.reduce_sum((y - y_pred)**2/n_samples, name='loss')
opt_op = tf.train.AdamOptimizer().minimize(loss)

summary_loss = tf.summary.scalar('loss', loss)

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('.', sess.graph)
    # Initialize Variables in graph
    sess.run(tf.global_variables_initializer())
    # Gradient descent loop for 500 steps
    for i in range(1000):
        # Select random minibatch
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = x_train[indices], y_train[indices]
        # Do gradient descent step
        _, loss_val, summary_loss_str = sess.run([opt_op, loss, summary_loss], feed_dict={x: X_batch, y: y_batch})
        file_writer.add_summary(summary_loss_str, i)

    file_writer.close()

    res = sess.run([W, b])

    w_res = res[0][0][0]
    b_res = res[0][0]

    x_line = np.array([0, 100])
    y_line = x_line * w_res + b_res

    # plot input data
    plt.plot(x_data, y_data)
    plt.plot(x_line, y_line)
    plt.show()


