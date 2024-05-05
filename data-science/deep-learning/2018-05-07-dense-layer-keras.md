---
slug: data-science/deep-learning//introduction-to-dense-net-with-keras/
title: Introduction to Dense Layers for Deep Learning with Keras
tags: [Data Science, Deep Learning, Dense Net, Iris, Keras]
---

The most basic neural network architecture in deep learning is the dense neural networks consisting of dense layers (a.k.a. fully-connected layers).

<!--truncate-->

In this layer, all the inputs and outputs are connected to all the neurons in each layer. Keras is the high-level APIs that runs on TensorFlow (and CNTK or Theano) which makes coding easier. Writing code in the low-level TensorFlow APIs is difficult and time-consuming. When I build a deep learning model, I always start with Keras so that I can quickly experiment with different architectures and parameters. Then, move onto TensorFlow to further fine tune it. When it comes to the first deep learning code, I think Dense Net with Keras is a good place to start. So, let’ get started.

Dataset

Deep learning 101 dataset is the classic MNIST, which is used for hand-written digit recognition. With the code below, you can certainly use MNIST.

In this example, I am using the machine learning classic Iris dataset. The dataset will be imported from a csv file. This gives you an idea on how to import csv into the deep learning model, rather than porting example data from the build-in package.

Deep learning on Iris certainly feels like cracking a nut with a sledge hammer. However, you can apply the knowledge and the same code to more appropriate datasets once you understand how it works.

There are many ways to get a csv version of Iris. I got it from R.

```bash
path = '/tmp/iris.csv'
data(iris)
write.csv(iris, path,row.names=FALSE)
```

Steps

(1) Import required modules

```python
import numpy as np
np.random.seed(21)
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
```

(2) Preprocessing

Both Keras and TensorFlow takes numpy arrays as features and classes. When the prediction is categorical, the outcome needs to be one-hot encoded (see one-hot encoding explanation from the Kaggle’s website). For one-hot encoding, the class needs to be indexes (starting from 0). Once they are transformed, you can use keras.utils.to_categorical() for conversion.

It uses sklearn.model_selection.train_test_split to create training and test dataset.

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

# Normalise features
features_norm = (features-np.min(features, axis=0)) / \
(np.max(features, axis=0)-np.min(features, axis=0))
print('Checking normalised features: \n{}'.format(features_norm[0:3]))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(features_norm, target, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_train[3])
print(y_train[3])

# Convert y into one hot encoded variables
n_classes = 3
y_train_e = keras.utils.to_categorical(y_train, n_classes)
y_test_e = keras.utils.to_categorical(y_test, n_classes)
print(y_train_e[0:3])
```

(3) Design Networks

I am using the sequential model with 2 fully-connected layers. ReLU is more popular in many deep neural networks, but I am using Tanh for activation because it actually performed better. You almost never use Sigmoid because it is slow to train. Softmax is used for the output layer.

Adding the 3rd layer degrades the performance. This makes sense as the data set is fairly simple. I am using Dropout to reduce over-fitting. L2 regularizer can be used. But, it did not perform well in this case and I commented out the line.

```python
model = Sequential()
# for l2 regularizer. this doesn't perform as good as dropouts
# model.add(Dense(32, activation='relu', input_dim=4, kernel_regularizer=regularizers.l2(0.01)))
# tanh is better than relu in this case
model.add(Dense(32, activation='tanh', input_shape=(4,)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
```

(4) Model Compilation

You need to define the loss function, optimizer and evaluation metrics. Cross-entropy is the gold standard for the cost function. You will almost never use quadratic. On the other hand, there are many options for optimisers. In this example, I have Adam as well as SGD with learning rate of 0.01. Both works fine.

```python
# Compared to mean_squared_error, cross entropy does faster learning,
# model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])
# We can use SGD with a specific learning rate for optimizer
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

(5) Execution

The testing accuracy goes up to 96.7% after 120 epochs. With this dataset, a regular machine learning algorithm like random forest or logistic regression can achieve the similar results. The first rule of deep learning is that if the simpler machine learning algorithm can achieve the same outcome, use machine learning and look for a more complicated problem. Here, the purpose is to learn the actual programming process so that you can apply it to more complex problems.

```python
epoch_n = 1000
model.fit(X_train, y_train_e, batch_size=60, epochs=epoch_n, verbose=1,\
 validation_data=(X_test, y_test_e))
```

Next Step

(1) Try using MNIST dataset on this code.

MNIST is included in Keras and you can imported it as keras.datasets.mnist. It’s already split into training and test datasets. In preprocessing, you need to flatten the data (from 28 x 28 to 784) and convert y into one-hot encoded values. Here is the code to process the data.

(2) Replicate the same code with low-level TensorFlow code.

TenorFlow is much more complicated than Keras. The way to code is quite unique. It will be difficult at first, but it will be worthwhile.

For the actual code example, go to Introduction to Dense Net with TensorFlow.
