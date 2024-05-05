---
slug: data-science/deep-learning//introduction-to-dense-net-with-tensorflow
title: Introduction to Dense Layers for Deep Learning with TensorFlow
tags: [Data Science, Deep Learning, Dense Net, Iris, TensorFlow]
---

TensorFlow offers both high- and low-level APIs for Deep Learning. Coding in TensorFlow is slightly different from other machine learning frameworks.

<!--truncate-->

You first need to define the variables and architectures. This is because the entire code is executed outside of Python with C++ and the python code itself is just a bunch of definitions.

The aim of this post is to replicate the previous Keras code into TensorFlow. Before writing code in TensorFlow, it is better to use high-level APIs like Keras to build the model (read Introduction to Dense Net with Keras for a preparation).

Steps

(1) Import Modules

```python
import tensorflow as tf
tf.set_random_seed(42)
tf.reset_default_graph()
import pandas as pd
import numpy as np
np.random.seed(42)
from sklearn.model_selection import train_test_split
import keras
```

(2) Data Preparation

As in the previous post, we are importing the Iris dataset from a csv file. This step is the same as before.

You can also check the official TensorFlow documents about deep learning on Iris dataset here. The way the dataset is preprocessed is quite different from what I did and will be an interesting read.

```python
iris = pd.read_csv('/tmp/iris.csv')
print(iris.head(5))
print(iris.Species.unique())
iris['Species'] = np.where(iris['Species']=='setosa', 0,
               np.where(iris['Species']=='versicolor', 1,
                       np.where(iris['Species']=='virginica', 2, 3)))
print(iris[0:3])
print(iris.Species.unique())

# convert df to features and targets
feature_cols = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
features = iris[feature_cols].values
target = iris.Species.values

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(X_train[3])
print(y_train[3])

# Convert y into one hot encoded variables
n_classes = 3
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

print(y_train[0:3])
```

(3) Defining Variables and Models

Before running the code, we need to define variables and models. The model is the same as the one defined in the previous post with Keras.

Even for more complicated models (e.g. with added convolutional layers), you can use the same steps.

Set hyperparameters
Set layers
Define placeholders
Define layers
Define architecture
Define variable dictionary
Build Model
Define loss & optimizer
Define evaluation metrics
Here is the code from the steps above.

```python
# (1) Set hyperparameters
lr = 0.01
epochs = 1000
batch_size = 20
weight_initializer = tf.contrib.layers.xavier_initializer()

# (2) Set layers
n_input = 4
n_dense = 10
n_classes = 3

# (3) Define placeholders
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# (4) Define layers
def dense(x, w, b):
    z = tf.add(tf.matmul(x, w), b)
    a = tf.nn.relu(z)
    return a

# (5) Define architecture
def network(x, weights, biases):
    dense1 = dense(x, weights['w1'], biases['b'])
    out_layer_z = tf.add(tf.matmul(dense1, weights['w_out']), biases['b_out'])
    return out_layer_z

# (6) Define variable dictionary
bias_dict = {
    'b': tf.Variable(tf.zeros([n_dense])),
    'b_out': tf.Variable(tf.zeros([n_classes]))
}

weight_dict = {
    'w1': tf.get_variable('w1', [n_input, n_dense], initializer = weight_initializer),
    'w_out': tf.get_variable('w_out', [n_dense, n_classes], initializer = weight_initializer)
}

# (7) Build Model
predictions = network(x, weights=weight_dict, biases=bias_dict)
print(predictions)

# (8) Define loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# (9) Define evaluation metrics
correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy_pct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
```

(4) Initialise and Run

Once everything is set up, initialise variables and execute the code in session! After 1000 epochs, you will see the test accuracy of 96%.

```python
# (10) Initialization
initializer_op = tf.global_variables_initializer()

# (11) Train the network in a session
with tf.Session() as session:
    session.run(initializer_op)

    print("Training for", epochs, "epochs.")

    # Go through epochs
    for epoch in range(epochs):
        # monitoring each epochs on training cost and accuracy
        avg_cost = 0.0
        avg_acc = 0.0

        # loop over all batches of the epoch:
        n_batches = int(120 / batch_size)
        for i in range(n_batches):

            # Get the random int for batch
            random_indices = np.random.randint(120, size=batch_size) # 120 is the no of training set records

            feed = {
                x: X_train[random_indices],
                y: y_train[random_indices]
            }

            # feed batch data to run optimization and fetching cost and accuracy:
            _, batch_cost, batch_acc = session.run([optimizer, cost, accuracy_pct],
                                                   feed_dict=feed)

            # accumulate mean loss and accuracy over epoch:
            avg_cost += batch_cost / n_batches
            avg_acc += batch_acc / n_batches

        # Training cost and accuracy at end of each epoch of training:
        print("Epoch ", '%03d' % (epoch+1),
              ": cost = ", '{:.3f}'.format(avg_cost),
              ", accuracy = ", '{:.2f}'.format(avg_acc), "%",
              sep='')

    print("Training Complete. Testing Model.\n")

    test_cost = cost.eval({x: X_test, y: y_test})
    test_acc = accuracy_pct.eval({x: X_test, y: y_test})

    print("Test Cost:", '{:.3f}'.format(test_cost))
    print("Test Accuracy: ", '{:.2f}'.format(test_acc), "%", sep='')
```
