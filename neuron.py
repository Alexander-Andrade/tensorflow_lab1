import tensorflow as tf


x = tf.constant(1.0, name='input')
w = tf.Variable(0.99, name='weight')
y = tf.multiply(w, x, name="output")
correct_y = tf.constant(0.0, name="correct_y")
loss = tf.pow(y - correct_y, 2, name='loss')
# loss = tf.losses.mean_squared_error(labels=correct_y, predictions=y)
opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

for value in [x, w, y, correct_y, loss]:
    tf.summary.scalar(value.op.name, value)

summaries = tf.summary.merge_all()

sess = tf.Session()
file_writer = tf.summary.FileWriter('.', sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(100):
    file_writer.add_summary(sess.run(summaries), i)
    sess.run(opt)

file_writer.close()
