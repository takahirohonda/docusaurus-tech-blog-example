---
slug: data-science/machine-learning/predict-internet-popularity-by-optimising-neural-networks-with-python
title: Predict Internet Popularity By Optimising Neural Networks With Python
tags:
  [
    Data Science,
    Machine Learning,
    caret,
    Hyper Parameter Search,
    Neural Networks,
    nnet,
    Python,
  ]
---

In the previous post, we used grid search to find the best hyper parameter for the neural network model with R’s caret package. Here, let’s use Python and scikit-learn package to optimise a neural network model.

Just like the caret package, scikit-learn has a pre-built function for hyper parameter search. As for dataset, we will use Online News Popularity Data Set from the UCI Machine Learning repository, which is the same dataset used in the previous post. It was originally used in this publication.

The nnet package from previous post is a single hidden layer back-propagation network. Therefore, there is only one size parameter for the layer to tune. Here, we will use MLPClassifier that implements a multi-layer perceptron algorithm. Having a multiple hidden layers enables us to specify the number of layers as well as the number of neurons for each layer.

As for parameter optimisation, we will use GridSearchCV with 10-fold cross validation. Optimising multi-layer neural networks can be lengthy because you can try different layer numbers and neuron size in each layer. Here, let’s use 2 layers and try to optimise the neuron size per layer for simplicity. Generally speaking, stating small works fine with neural networks in terms of both layer and neuron sizes.

We will also search for the best alpha parameter for L2 regularisation (you can read more about regularisation here). In short, regularisation prevents over-fitting by penalising it. With no regularisation term, you will get a great accuracy on training set, but not so great on test. Neural networks are often called as a black box method, which makes the method fancy and magical. In realisty, they are a variation of nonlinear statistical models. So, regularisation terms work just like logistic regression. Scikit-learn has a great documentation on MPL here for more detail.

As in the previous post, we are dealing with the prediction of binary classifier (‘popular’ or ‘unpopular’) based on the attributes comes with online news paper articles (see details here).

OK, let’s code!

Summary Steps

Get data and prep it (by selecting the right columns, splitting them to training and test and normalising the data).
Do hyperparameter search.
Model & predict with the best hyper parameter.
Calculate performance metrics and draw ROC curve.
Code

(1) First of all, we have to load the data (after downloading data from here) and do data prep. I normalised it after splitting into training and test set, but you can do it prior to the split. Note that you have to use the scaler function fitted with training data for the test data set to keep consistency if you are doing this way.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# (1) Get data
shares = pd.read_csv("/tmp/OnlineNewsPopularity.csv")

# (2) Check data
shares.head(5)
shares.shape

# (3) Prepare data (train_test split and normalise)
X = shares.iloc[:,2:59]
y = shares.iloc[:,60]

y = np.where(y >= 1400, 'Popular', 'Unpopular')

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(train_x)
# Now apply the transformations to the data:
train_x = scaler.transform(train_x)
test_x = scaler.transform(test_x)
```

(2) I created a grid_search function so that I can reuse it for other models. It takes a scikit-learn model, feature matrix, label array and parameter grids and return the best parameters.

```python
from sklearn.model_selection import GridSearchCV

def grid_search(estimator, X, y, param_grid):
    '''Takes a sklearn model, feature matrix, label array and parameter grid dictionary.
    Returns the dictionary of best parameters'''
    clf = GridSearchCV(estimator, param_grid = param_grid, n_jobs=-1, cv=10, scoring='f1_weighted')
    clf.fit(X, y)
    print('Best F1 Score is: {}'.format(clf.best_score_))
    print('Best Parameter is: {}'.format(clf.best_params_))
    return clf.best_params_
```

(3) See the range of the alpha and neuron size for each layer that I choose. For some reason, neuron size (5, 2) appear in many places (including the official documentation). So, I chose my parameter grid range around them. The bigger the neuron size is, the more complex the models become. Generally speaking, a smaller neuron size works for most of the time. But, of course, you have to experiment it a lot to find the best size. You can read this post for more details. It is a good starting point to know about layer and neuron size.

I picked lbfgs for a solver. You can try to use different ones. Again, you have to experiment it to see which one works best for your use case.

As I did with R code here, you can do parallel computing by adding the n_jobs=-1. As the grid search is an iterative CPU intense process, it is always faster to do parallel.

```python
from sklearn.neural_network import MLPClassifier

alpha = [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3, 1e-2,3e-2]
hidden_layer_sizes = [(3,1), (5,2), (9,4)]
param_grid = {'alpha':alpha, 'hidden_layer_sizes':hidden_layer_sizes}
estimator = MLPClassifier(solver='lbfgs',random_state=1)
best_param = grid_search(estimator, train_x, train_y, param_grid)
```

Best F1 Score is: `0.6631095339771439`
Best Parameter is: `{‘alpha’: 3e-05, ‘hidden_layer_sizes’: (5, 2)}`

(4) Now, we got the best hyperparameter set, let’s model the neural net and do prediction.

```python
clf = MLPClassifier(solver='lbfgs', alpha=best_param['alpha'],\
        hidden_layer_sizes=best_param['hidden_layer_sizes'], random_state=1)
clf.fit(train_x, train_y)
pred_train = clf.predict(train_x)
pred_test = clf.predict(test_x)

pred_train_prob = clf.predict_proba(train_x)[:, 1]
pred_test_prob = clf.predict_proba(test_x)[:, 1]
```

(5) I created two functions to get all the evaluation metrics. The calculate_auc function also produces ROC. I used pandas magic to create a data frame for the easy summary of performance metrics. These metrics are the same one use in the publication.

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
%matplotlib inline

def evaluation(train_y, pred_train, test_y, pred_test, model_name):

    train_acc = accuracy_score(train_y, pred_train)
    test_acc = accuracy_score(test_y, pred_test)

    train_precision = precision_score(train_y, pred_train, pos_label='Unpopular')
    test_precision = precision_score(test_y, pred_test, pos_label='Unpopular')

    train_recall = recall_score(train_y, pred_train, pos_label='Unpopular')
    test_recall = recall_score(test_y, pred_test, pos_label='Unpopular')

    train_f1 = f1_score(train_y, pred_train, pos_label='Unpopular')
    test_f1 = f1_score(test_y, pred_test, pos_label='Unpopular')

    train_df = pd.DataFrame({'Accuracy':[train_acc], 'Precision':[train_precision],\
                             'Recall':[train_recall], 'F1':[train_f1]})
    test_df = pd.DataFrame({'Accuracy':[test_acc], 'Precision':[test_precision],\
                             'Recall':[test_recall], 'F1':[test_f1]})
    return train_df, test_df

traindf1, testdf1 = evaluation(train_y, pred_train, test_y, pred_test, 'Neural Network')

def calculate_auc(train_y, pred_train_prob, test_y, pred_test_prob, model_name):
    fpr_train, tpr_train, thresholds_train = roc_curve(train_y, pred_train_prob, pos_label='Unpopular')
    fpr_test, tpr_test, thresholds_test = roc_curve(test_y, pred_test_prob, pos_label='Unpopular')
    train_auc = auc(fpr_train, tpr_train)
    test_auc = auc(fpr_test, tpr_test)

    # Draw ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr_test, tpr_test, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % test_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Data ROC Curve With Neural Network')
    plt.legend(loc="lower right")
    plt.show()

    train_df = pd.DataFrame({'AUC':[train_auc]})
    test_df = pd.DataFrame({'AUC':[test_auc]})
    return train_df, test_df

traindf2, testdf2 = calculate_auc(train_y, pred_train_prob, test_y, pred_test_prob, 'Neural Network')

train_df = pd.concat([traindf1, traindf2], axis=1)
test_df = pd.concat([testdf1, testdf2], axis=1)

print("Training Data Performance Metrics\n")
print(round(train_df, 2))
print("\nTest Data Performance Metrics\n")
print(round(test_df, 2))
```

Here is the output.

![ROC Curve](./img/roc-for-neural-net-with-python.webp)

Comparing it With Default MLP Classifier Model

If you don’t set any parameter, the default alpha is 0.0001 and hidden_layer_sizes is 100 neurons in a single layer. The default model is over-fitting, which happens a lot for neural networks. You can see the power of simple parameter tuning!

```python
clf = MLPClassifier(solver='lbfgs', random_state=1) # default alpha=1e-5
clf.fit(train_x, train_y)
pred_train = clf.predict(train_x)
pred_test = clf.predict(test_x)
train_acc = accuracy_score(train_y, pred_train)
test_acc = accuracy_score(test_y, pred_test)
print('Neural Network Model Train Accuracy: {}'.format(train_acc))
print('Neural Network Model Test Accuracy: {}'.format(test_acc))
```

Neural Network Model Train Accuracy: 0.77
Neural Network Model Test Accuracy: 0.62

Let’s Benchmark!

Below is the performance metric table from the paper (Fernandes et al. 2015). Our neural net model was comparable to the second best method (AdaBoost) in terms of accuracy and AUC. The model is still over-fitting slightly and has lower recall. So, I think there are more room for optimisation. All in all, pretty good outcome considering how simple the whole modelling process was.

![Benchmark](./img/performance-benchmark-python.webp)

Your Turn

In the publication, they also used grid search to find the best hyper parameter set in a few different methods. In the paper, SCV was optimised with C ∈ {2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6} , RF and AdaBoost was with number of trees ∈ {10, 20, 50, 100, 200, 400} and KNN was number of neighbors ∈ {1, 3, 5, 10, 20}. Now that we have a custom grid_search function, you can try finding the best hyper parameters for all these methods in a streamlined manner.

```python
from sklearn.svm import SVC
svc_model = SVC(kernel='rbf', random_state=23)
svc_param = {'C':[2**0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6]}
xx, yy, zz = grid_search(svc_model, train_x, train_y, svc_param)

from sklearn.ensemble import RandomForestClassifier
# n_estimator is the number of trees
rf_param = {'n_estimators':[10, 20, 50, 100, 200, 400]}
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
xx, yy, zz = grid_search(rf, train_x, train_y, rf_param)

from sklearn.neighbors import KNeighborsClassifier
knn_param = {'n_neighbors':[10, 20, 50, 100, 200, 400]}
knn = KNeighborsClassifier(n_jobs=-1)
xx, yy, zz = grid_search(knn, train_x, train_y, knn_param)
```
