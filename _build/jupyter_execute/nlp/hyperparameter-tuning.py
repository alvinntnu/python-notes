#!/usr/bin/env python
# coding: utf-8

# # Hyper-Parameter Tuning
# 
# 

# There are two important techniques to fine-tune the hyperparameters of the model: Grid Search and Cross Validation.
# 
# - Grid Search
#   - Define a few parameter values and experiment all these values in modeling. Use `sklearn.model_selection.GridSearchCV` to find the best parameter settings.
# - Cross Validation
#   - Fine-tune the parameters using cross-validation. Common CV methods include `sklearn.model_selection.StratifiedKFold`, `sklearn_model_selection.ShuffleSplit`, `LeaveOneOut`.
# 
# Both mothods can make use of the `pipeline` in sklearn to streamline the processing of training and validation.

# ## SVM Model
# 
# - This example is from the officient `sklearn` documentation
# - A classic SVM model with train and test split

# In[1]:


## Normal SVM model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape

## fit on train sets
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
## test on test sets
clf.score(X_test, y_test)


# ## Default K-fold Cross Validation
# 
# - We can easily use the `cross_val_score()` to do the K-fold cross validation for a specific training model (without shuffling)
# 
# :::{note}
# By default, `train_test_split` returns a random split.
# :::

# In[2]:


## Cross-validation

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=10, scoring='f1_macro')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



# ## GridSearchCV and Fine-Tuning Hyper-Parameters
# 
# - Important steps:
#   - Define SVM classifier
#   - Define a set of parameter values to experiment
#   - Use `GridSearchCV` to find the best parameter settings
#   - `GridSearchCV` by default implements cross-validation methods to find the optimal parameter settings.
#   - In other words, we can specify our own more sophisticated CV methods in `GridSearchCV`.

# In[3]:


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5) # 
clf.fit(iris.data, iris.target)


sorted(clf.cv_results_.keys())

#print(clf.cv_results_)

import pandas as pd

cv_results_df=pd.DataFrame(clf.cv_results_)
cv_results_df


# ## Cross-validation, Hyper-Parameter Tuning, and Pipeline

# - Common cross validation methods:
#   - `StratifiedKFold`: Split data into train and validation sets by preserving the percentage of samples of each class
#   - `ShuffleSplit`: Split data into train and validation sets by first shuffling the data and then splitting
#   - `StratifiedShuffleSplit`: Stratified + Shuffled
#   - `LeaveOneOut`: Creating train sets by taking all samples execept one, which is left out for validation
# - Important Steps
#   - Define all preprocessing methods as Tranasformer
#   - Create a pipeline
#   - Define a cross-validation method
#   - Use `cross_val_score()` to fine the best parameter settings by feeding the pipeline and the CV method

# In[4]:


from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit

## Create pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))

## Define CV methods
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
shuffsplit = ShuffleSplit(n_splits=5, test_size=0.1, random_state=7)

## CV results
print('K-fold CV:', cross_val_score(clf, X, y, cv=kfold))
print('Shuffle CV:', cross_val_score(clf, X, y, cv=shuffsplit))


# ## Deep Learning Example
# 
# - This is based on [GridSearchCV with keras](https://www.kaggle.com/shujunge/gridsearchcv-with-keras)

# ## MNIST Dataset

# In[5]:


import numpy as np
import os
from keras.datasets import mnist
from keras.layers import *
from keras.models import *
from time import time
nb_classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[6]:


## Convert y labels (integers) into binary class matrix
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print(Y_train[:5,])


# ## Hyper-Parameter Tuning Using Grid Search

# In this example, we aim to fine-tune the following hyper-parameters of the deep neural network:
# 
# - `optimizer`
# - `kernel_initializer` of the Dense layer
# 
# We can fine-tune other common parameters like epochs and batch-sizes.

# In[7]:


def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
    model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dense(512, kernel_initializer=init))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer=init))
    model.add(Activation('softmax')) # This special "softmax" a
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
    return model


# In[8]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
start=time()
model = KerasClassifier(build_fn=create_model)
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
# epochs = [50, 100, 150]
# batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, Y_train)
print("total time:",time()-start)


# In[9]:


## The Best Model
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

## Print all experiment results
import pandas as pd
results = pd.DataFrame(grid_result.cv_results_)
results


# ## K-fold Validation

# - Similar to shuffle split?

# In[10]:


np.random.seed(7)
seed=12


# In[11]:


def create_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
    model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax')) # This special "softmax" a
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 
    return model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
start=time()

model2 = KerasClassifier(build_fn=create_model, epochs=10, batch_size=1500,verbose=1)
results = cross_val_score(model2, X_train, Y_train, cv=10)

print("K-fold Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

print("total time:",time()-start)




# ## Shuffle Split Cross Validation
# 
# - Shuffle but not stratified

# In[12]:


## Shuffle Split
start=time()
sf = ShuffleSplit(n_splits=10)
results = cross_val_score(model2, X_train, Y_train, cv=sf)
print("K-fold Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("total time:",time()-start)


# ## Stratified K-Fold Validation
# 

# - Stratified but not shuffled
# - Stratified cross-validation gives poor results. No idea why? 

# In[13]:


print(Y_train[:4,])
print(y_train[:4])
from collections import Counter

Counter(y_train)


# In[14]:


## Stratified Shuffle
from sklearn.model_selection import StratifiedKFold
start=time()
sk = StratifiedKFold(n_splits=10)
results = cross_val_score(model2, X_train, y_train, cv=sk)
print("Stratified K-fold Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("total time:",time()-start)


# ## Stratified Shuffle Split
# 
# - A mix of `StratifiedKFold` and `ShuffleSplit`, which returns stratified randomized folds
# - The folds are made by preserving the percentage of samples for each class.

# In[15]:


## Stratified Shuffle
from sklearn.model_selection import StratifiedShuffleSplit
start=time()
ss = StratifiedShuffleSplit(n_splits=10)
results = cross_val_score(model2, X_train, Y_train, cv=ss)
print("Stratified Shuffle Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("total time:",time()-start)


# ## GridSearchCV and Cross-Validation
# 
# - We perform the `GridSearchCV` to find the optimal optimizers
# - We determine the best parameter values based on stratified shuffled 5-fold cross validation

# In[16]:


def create_model():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
    model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax')) # This special "softmax" a
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 
    return model
start=time()

## Define cross-validation method
ss = StratifiedShuffleSplit(n_splits=5)

## Define Model
model3 = KerasClassifier(build_fn=create_model)

## Define Hyper-parameter values
optimizers = ['rmsprop', 'adam']
param_grid = dict(optimizer=optimizers, init=init)

## Define GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=ss)
grid_result = grid.fit(X_train, Y_train)
print("total time:",time()-start)


# In[17]:


## The Best Model
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

## Print all experiment results
import pandas as pd
results = pd.DataFrame(grid_result.cv_results_)
results


# ## References
# 
# - [`GridSearchCV` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
# - [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
# - [Scoring Parameters](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
# - [GridSearchCV with keras](https://www.kaggle.com/shujunge/gridsearchcv-with-keras)

# ## Requirements

# In[18]:


# %run ./get_modules.py
import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name
        
        
imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))

