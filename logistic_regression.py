import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


# Step 1: Read the data
iris = pd.read_csv('iris.csv')
print(iris.shape)
print(iris.head())

# I want to do a binary classification, so keep the first 100 rows of data
# Iris-setosa species is linearly separable from the other two,
# but the other two are not linearly separable from each other.
# To keep the species blance Iris-setosa and Iris-versicolor are choosen
iris = iris[:100]
print(iris.shape)

f, axes = plt.subplots(2, 2)

# Step 2: Numerical processing
# replace 'Iris-setosa' as 0
# replace 'Iris-versicolor' as 1
iris.Species = iris.Species.replace(to_replace=['Iris-setosa', 'Iris-versicolor'], value=[0, 1])
axes[0][0].scatter(iris[:50].SepalLengthCm, iris[:50].SepalWidthCm, label='Iris-setosa')
axes[0][0].scatter(iris[51:].SepalLengthCm, iris[51:].SepalWidthCm, label='Iris-versicolor')
axes[0][0].set_xlabel('SepalLength')
axes[0][0].set_ylabel('SepalWidth')
axes[0][0].legend(loc='best')

X = iris.drop(labels=['Id', 'Species'], axis=1).values
y = iris.Species.values

# Step 3: Split data
# trainset: 80%
# testset: 20%

# set replace=False, Avoid double sampling
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)

# diff set
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]


# Define the normalized function
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


# Step 4: Normalized processing
# Normalized processing, must be placed after the data set segmentation,
# otherwise the test set will be affected by the training set
train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)

# Step 5: Build the model framework
# Begin building the model framework
# Declare the variables that need to be learned and initialization
# There are 4 features here, A's dimension is (4, 1)
A = tf.Variable(tf.random_normal(shape=[4, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Define placeholders
data = tf.placeholder(dtype=tf.float32, shape=[None, 4])
target = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Declare the model you need to learn
model = tf.matmul(data, A) + b

# Declare loss function
# Use the sigmoid cross-entropy loss function,
# first doing a sigmoid on the model result and then using the cross-entropy loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=target))

# Define the learning rate， batch_size etc.
learning_rate = 0.003
batch_size = 30
iter_num = 1500

# Define the optimizer
opt = tf.train.GradientDescentOptimizer(learning_rate)

# Define the goal
goal = opt.minimize(loss)

# Define the accuracy
# The default threshold is 0.5, rounded off directly
prediction = tf.round(tf.sigmoid(model))
# Bool into float32 type
correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
# Average
accuracy = tf.reduce_mean(correct)
# End of the definition of the model framework

# Start training model
# Define the variable that stores the result
loss_trace = []
train_acc = []
test_acc = []


# Step 6: Model training
# training model
for epoch in range(iter_num):
    # Generate random batch index
    batch_index = np.random.choice(len(train_X), size=batch_size)
    batch_train_X = train_X[batch_index]
    batch_train_y = np.matrix(train_y[batch_index]).T
    sess.run(goal, feed_dict={data: batch_train_X, target: batch_train_y})
    temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
    # convert into a matrix, and the shape of the placeholder to correspond
    temp_train_acc = sess.run(accuracy, feed_dict={data: train_X, target: np.matrix(train_y).T})
    temp_test_acc = sess.run(accuracy, feed_dict={data: test_X, target: np.matrix(test_y).T})
    # recode the result
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    # output
    if (epoch + 1) % 300 == 0:
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} test_acc: {:5f}'.format(epoch + 1, temp_loss,
                                                                          temp_train_acc, temp_test_acc))


# Step 7: Visualization
# Visualization of the results
# loss function
axes[0][1].plot(loss_trace)
axes[0][1].set_title('Cross Entropy Loss')
axes[0][1].set_xlabel('epoch')
axes[0][1].set_ylabel('loss')

# accuracy
axes[1][0].plot(train_acc, 'b-', label='train accuracy')
axes[1][0].plot(test_acc, 'k-', label='test accuracy')
axes[1][0].set_xlabel('epoch')
axes[1][0].set_ylabel('accuracy')
axes[1][0].set_title('Train and Test Accuracy')
axes[1][0].legend(loc='best')
plt.show()
