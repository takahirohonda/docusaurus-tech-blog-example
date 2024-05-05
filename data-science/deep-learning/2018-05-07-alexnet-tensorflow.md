---
slug: data-science/deep-learning//building-alexnet-with-tensorflow-and-running-it-with-aws-sagemaker
title: Building AlexNet with TensorFlow and Running it with AWS SageMaker
tags:
  [
    Data Science,
    Deep Learning,
    AlexNet,
    Convolutional Neural Networks,
    Image Classification,
    SageMaker,
    TensorFlow,
  ]
---

In the last post, we built AlexNet with Keras. This is the second part of AlexNet building. Let’s rewrite the Keras code from the previous post (see Building AlexNet with Keras) with TensorFlow and run it in AWS SageMaker instead of the local machine.

<!--truncate-->

AlexNet is in fact too heavy for a regular commercial laptop to handle it. It only runs on a small dataset and takes for ages. By using the cloud service like AWS, we can access to much better computers without any hardware investment. AWS already has a series of deep learning specialised instances (P2 Instances). The smallest with one GPU (p2.xlarge) costs 90 cent per hour. It is much faster than CPU machines. You can experiment on computing capacities as you will be charged only by usage hours. If you are thinking about buying a more expensive GPU laptop for deep learning, the cloud services would be a better option.

AWS recently released SageMaker, which enables you to develop and deploy deep learning code with no hustle. To run Tensorflow code in SageMaker, all you need is to create a notebook instance (check out the getting started video here).

**Importing OxfordFlower17 Data**

You can creates a notebook instance with a chosen EC2 instance with SageMaker. Once the instance is created, you can access to the instance through Jupyter notebook for development and deployment. Many deep learning frameworks are already installed. Once you train the model, you can deploy it into the AWS environment without much hustle. The caveat is that you won’t be able to install or update the preinstalled packages as you do not have access to the underlining instance. If you need to have special environmental requirements, you need to bring it in with a Docker container.

In fact, SageMaker does not have tflearn installed. As in the previous post, we are importing 17 category flower dataset (OxfordFlower17) from tflearn. If you try to import it in SageMaker, it will give you the module not found error.

The strategy I took here is to upload the dataset as numpy array files to S3 and retrieve them in SageMaker.

(1) Create the numpy files and Upload to S3

I first created npy files and uploaded to S3 bucket where SageMaker has the access policy.

```python
# (1) Get dataset from s3 bucket
import tflearn.datasets.oxflower17 as oxflower17
x, y = oxflower17.load_data(one_hot=True)

# (2) save data as .npy files
x_path='/tmp/oxford_flower_17_x.npy'
y_path='/tmp/oxford_flower_17_y.npy'
import numpy as np
np.save(x_path, x)
np.save(y_path, y)

# (3) push file to s3
import boto3
import botocore

client = boto3.client('s3')
target_bucket = 'sagemaker.mydh'
x_key = 'data/oxford_flower_17_x.npy'
y_key = 'data/oxford_flower_17_y.npy'

# Uploading
client.upload_file(Filename=x_path, Bucket=target_bucket, Key=x_key)
print('Completed Uploading {} to {}/{}'.format(x_path, target_bucket, x_key))
client.upload_file(Filename=y_path, Bucket=target_bucket, Key=y_key)
print('Completed Uploading {} to {}/{}'.format(y_path, target_bucket, y_key))
```

(2) Import numpy files into the SageMaker instance.

You can get the file from S3 into the Notebook instance and simply load them as numpy objects.

```python
import boto3
import botocore

def download_file_with_resource(bucket_name, key, local_path):
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).download_file(key, local_path)
    print('Downloaded {}'.format(key))

bucket_name = 'sagemaker.mydh'
x_key='data/oxford_flower_17_x.npy'
y_key='data/oxford_flower_17_y.npy'
x_local_path = './data/oxford_flower_17_x.npy'
y_local_path = './data/oxford_flower_17_y.npy'

# download_file_with_resource(bucket_name, 'data/check.csv', './data/check.csv')
download_file_with_resource(bucket_name, x_key, x_local_path)
download_file_with_resource(bucket_name, y_key, y_local_path)

# (3) Load data
x_local_path = './data/oxford_flower_17_x.npy'
y_local_path = './data/oxford_flower_17_y.npy'
x = np.load(x_local_path)
y = np.load(y_local_path)
print('Shape of features: ', x.shape, 'Type: ', type(x))
print('Shape of classes: ', y.shape, 'Type: ', type(y))
```

Code

Strictly speaking, it is slightly different from the original AlexNet. The code is sequential and has no parallel computing components for simplicity. I am doing batch normalisation before every input and doing dropouts in the Dense layer. The network architecture is the same as the previous post.

With TensorFlow, you really need to be careful about the dimensions. The original dataset is 3-dimentional. After the convolution layers, the dimension is compressed from pooling. So, you need to specify the right dimension (7 x 7 in this case). Otherwise, the code will not run.

In the model, I purposely included the weights and biases with hard-coded values so that it is easy to follow. Apart from the model, the same code used in building Dense Net for Iris works. If you need to understand other part of the codes you should read the previous post (Introduction to Dense Net with TensorFlow).

```python
import tensorflow as tf
tf.set_random_seed(1000)
import numpy as np
np.random.seed(1000)
from sklearn.model_selection import train_test_split

# (1) Create Training (80%), test (20%) and validation (20%) dataset
#     Datasets (x and y) are loaded as numpy object from the previous step
x_train, x_test_pre, y_train, y_test_pre = train_test_split(x, y, test_size=0.20, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test_pre, y_test_pre, test_size=0.1)

# Check Shapes
print('Shape: x_train={}, y_train={}'.format(x_train.shape, y_train.shape))
print('Shape: x_test={}, y_test={}'.format(x_test.shape, y_test.shape))
print('Shape: x_validation={}, y_validation={}'.format(x_validation.shape, y_validation.shape))

# (2) Define the placeholder tensors
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 17]) # no of flower speces in the dataset

# (3) Define Layers
# Convolutional Layer with Relu activation
def conv2D(x, W, b, stride_size):
    xW = tf.nn.conv2d(x, W, strides=[1, stride_size, stride_size, 1],padding='SAME')
    z = tf.nn.bias_add(xW, b)
    a = tf.nn.relu(z)
    return (a)

# Max Pooling Layer
def maxPooling2D(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                         strides=[1, stride_size, stride_size, 1],padding='SAME')

# Dense Layer
def dense(x, W, b):
    z = tf.add(tf.matmul(x, W), b)
    a = tf.nn.relu(z)
    return a

# (4) Define AlexNet
# Setting some parameters
w_init = tf.contrib.layers.xavier_initializer()
batch_size = 8
epochs = 1
progress = 40
n_classes = 17

# Function, x is the input features
def alexNet(img_input):

    # 1st Convolutional Layer
    w_c1 = tf.get_variable('w_c1', [11, 11, 3, 96], initializer=w_init)
    b_c1 = tf.Variable(tf.zeros([96]))
    c1 = conv2D(img_input, w_c1, b_c1, stride_size=4)
    # Pooling
    p1 = maxPooling2D(c1, kernel_size=2, stride_size=2)
    # Batch Normalisation
    bn1 = tf.contrib.layers.batch_norm(p1)

    # 2nd Convolutional layer
    w_c2 = tf.get_variable('w_c2', [5, 5, 96, 256], initializer=w_init)
    b_c2 = tf.Variable(tf.zeros([256]))
    c2 = conv2D(bn1, w_c2, b_c2, stride_size=1)
    # Pooling
    p2 = maxPooling2D(c2, kernel_size=2, stride_size=2)
    # Batch Normalisation
    bn2 = tf.contrib.layers.batch_norm(p2)

    # 3rd Convolutional Layer
    w_c3 = tf.get_variable('w_c3', [3, 3, 256, 384], initializer=w_init)
    b_c3 = tf.Variable(tf.zeros([384]))
    c3 = conv2D(bn2, w_c3, b_c3, stride_size=1)
    # Batch Normalisation
    bn3 = tf.contrib.layers.batch_norm(c3)

    # 4th Convolutional Layer
    w_c4 = tf.get_variable('w_c4', [3, 3, 384, 384], initializer=w_init)
    b_c4 = tf.Variable(tf.zeros([384]))
    c4 = conv2D(bn3, w_c4, b_c4, stride_size=1)
    # Batch Normalisation
    bn4 = tf.contrib.layers.batch_norm(c4)

    # 5th Convolutional Layer
    w_c5 = tf.get_variable('w_c5', [3, 3, 384, 256], initializer=w_init)
    b_c5 = tf.Variable(tf.zeros([256]))
    c5 = conv2D(bn4, w_c5, b_c5, stride_size=1)
    # Pooling
    p3 = maxPooling2D(c5, kernel_size=2, stride_size=2)
    # Batch Normalisation
    bn5 = tf.contrib.layers.batch_norm(p3)

    # Flatten the conv layer - features has been reduced by pooling 3 times: 224/2*2*2
    flattened = tf.reshape(bn5, [-1, 28*28*256])

    # 1st Dense layer
    w_d1 = tf.get_variable('w_d1', [28*28*256, 4096], initializer=w_init)
    b_d1 = tf.Variable(tf.zeros([4096]))
    d1 = dense(flattened, w_d1, b_d1)
    # Dropout
    dropout_d1 = tf.nn.dropout(d1, 0.6)

    # 2nd Dense layer
    w_d2 = tf.get_variable('w_d2', [4096, 4096], initializer=w_init)
    b_d2 = tf.Variable(tf.zeros([4096]))
    d2 = dense(dropout_d1, w_d2, b_d2)
    # Dropout
    dropout_d2 = tf.nn.dropout(d2, 0.6)

    # 3rd Dense layer
    w_d3 = tf.get_variable('w_d3', [4096, 1000], initializer=w_init)
    b_d3 = tf.Variable(tf.zeros([1000]))
    d3 = dense(dropout_d2, w_d3, b_d3)
    # Dropout
    dropout_d3 = tf.nn.dropout(d3, 0.6)

    # Output layer
    w_out = tf.get_variable('w_out', [1000, n_classes], initializer=w_init)
    b_out = tf.Variable(tf.zeros([n_classes]))
    out = tf.add(tf.matmul(dropout_d3, w_out), b_out)

    return out

# (5) Build model
predictions = alexNet(x)

# (6) Define model's cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# (7) Defining evaluation metrics
correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy_pct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

# (8) initialize
initializer_op = tf.global_variables_initializer()

# (9) Run
with tf.Session() as session:
    session.run(initializer_op)

    print("Training for", epochs, "epochs.")

    # looping over epochs:
    for epoch in range(epochs):
        # To monitor performance during training
        avg_cost = 0.0
        avg_acc_pct = 0.0

        # loop over all batches of the epoch- 1088 records
        # batch_size = 128 is already defined
        n_batches = int(1088 / batch_size)
        counter = 1
        for i in range(n_batches):

            # Get the random int for batch
            random_indices = np.random.randint(1088, size=batch_size) # 1088 is the no of training set records

            feed = {
                x: x_train[random_indices],
                y: y_train[random_indices]
            }

            # feed batch data to run optimization and fetching cost and accuracy:
            _, batch_cost, batch_acc = session.run([optimizer, cost, accuracy_pct],
                                                   feed_dict=feed)
            # Print batch cost to see the code is working (optional)
            # print('Batch no. {}: batch_cost: {}, batch_acc: {}'.format(counter, batch_cost, batch_acc))
            # Get the average cost and accuracy for all batches:
            avg_cost += batch_cost / n_batches
            avg_acc_pct += batch_acc / n_batches
            counter += 1

        # Get cost and accuracy after one iteration
        test_cost = cost.eval({x: x_test, y: y_test})
        test_acc_pct = accuracy_pct.eval({x: x_test, y: y_test})
        # output logs at end of each epoch of training:
        print("Epoch {}: Training Cost = {:.3f}, Training Acc = {:.2f} -- Test Cost = {:.3f}, Test Acc = {:.2f}"\
              .format(epoch + 1, avg_cost, avg_acc_pct, test_cost, test_acc_pct))

    # Getting Final Test Evaluation
    print('\n')
    print("Training Completed. Final Evaluation on Test Data Set.\n")
    test_cost = cost.eval({x: x_test, y: y_test})
    test_accy_pct = accuracy_pct.eval({x: x_test, y: y_test})
    print("Test Cost:", '{:.3f}'.format(test_cost))
    print("Test Accuracy: ", '{:.2f}'.format(test_accy_pct), "%", sep='')
    print('\n')

    # Getting accuracy on Validation set
    val_cost = cost.eval({x: x_validation, y: y_validation})
    val_acc_pct = accuracy_pct.eval({x: x_validation, y: y_validation})
    print("Evaluation on Validation Data Set.\n")
    print("Evaluation Cost:", '{:.3f}'.format(val_cost))
    print("Evaluation Accuracy: ", '{:.2f}'.format(val_acc_pct), "%", sep='')
```
