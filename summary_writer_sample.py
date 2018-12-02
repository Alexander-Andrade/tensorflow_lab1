import tensorflow as tf
import os


a = tf.constant(5.0)
b = tf.constant(6.0)

c = a * b

sess = tf.Session()
file_writer = tf.summary.FileWriter(os.getcwd(), sess.graph)
file_writer.close()

with tf.Session() as sess:
    print(sess.run(c))
