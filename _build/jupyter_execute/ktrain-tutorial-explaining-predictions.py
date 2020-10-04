# Explainable AI


- This notebook runs on Google Colab
- This is based on the `ktrain` official tutorial [Explainable Ai in krain](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A2-explaining-predictions.ipynb)

## Preparation on Colab

- Mount Google Drive
- Install `ktrain`
- Set the default `DATA_ROOT` (the root directory of the data files)

## Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

## Install ktrain
!pip install git+https://github.com/amaiya/eli5@tfkeras_0_10_1

## Working Directory

- Set the working directory to the `DATA_ROOT`

## Set DATA_ROOT
DATA_ROOT = '/content/drive/My Drive/_MySyncDrive/RepositoryData/data'
%cd '$DATA_ROOT'

## Check Working Directory
%pwd

## Autoreload

%reload_ext autoreload
%autoreload 2
%matplotlib inline
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

## Overview: Explainable AI in *ktrain*

- [**Explainable AI (XAI)**](https://www.darpa.mil/program/explainable-artificial-intelligence)
- Obtain the `Predictor` with `ktrain.get_predictor`
- In combination with text data, one can easily make predictions from the raw and unprocessed text of a document as follows:

```
predictor = ktrain.get_predictor(learner.model, preproc=preproc)
predictor.predict(document_text) 
```
- Utilize the `explain` method of `Predictor` objects to help understand how those predictions were **made**

## Three Types of Prediction

- Deep Learning Model Using BERT
- Classification
- Regression

## Sentiment Analysis Using BERT

### Data

- Marc's thesis data (Chinese movie reviews)

# imports
import ktrain
from ktrain import text

import pandas as pd

raw_csv = pd.read_csv("marc_movie_review_metadata.csv")
raw_csv = raw_csv.rename(columns={'reviews':'Reviews', 'rating':'Sentiment'})
raw_csv

from sklearn.model_selection import train_test_split

data_train, data_test = train_test_split(raw_csv, test_size=0.1)
## dimension of the dataset

print("Size of train dataset: ",data_train.shape)
print("Size of test dataset: ",data_test.shape)

#printing last rows of train dataset
data_train.tail()

#printing head rows of test dataset
data_test.head()

### Train-Test Split

# STEP 1: load and preprocess text data
# (x_train, y_train), (x_test, y_test), preproc = text.texts_from_df('aclImdb', 
#                                                                        max_features=20000, maxlen=400, 
#                                                                        ngram_range=1, 
#                                                                        train_test_names=['train', 'test'],
#                                                                        classes=['pos', 'neg'],
#                                                                        verbose=1)


(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=data_train,
                                                                   text_column = 'Reviews',
                                                                   label_columns = 'Sentiment',
                                                                   val_df = data_test,
                                                                   maxlen = 250,
                                                                   lang = 'zh-*',
                                                                   preprocess_mode = 'bert') # or distilbert

### Define Model

# # STEP 2: define a Keras text classification model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
# model = Sequential()
# model.add(Embedding(20000+1, 50, input_length=400)) # add 1 for padding token
# model.add(GlobalAveragePooling1D())
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# learner = ktrain.get_learner(model, train_data=(x_train, y_train), val_data=(x_test, y_test))


# # STEP 2: define a text classification model
## use 'distilbert' if you want
model = text.text_classifier(name = 'bert', # or distilbert
                             train_data = (X_train, y_train),
                             preproc = preproc)
#here we have taken batch size as 6 as from the documentation it is recommend to use this with maxlen as 500
learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),
                   val_data = (X_test, y_test),
                   batch_size = 6)

### Fitting Model


## STEP 3: train
## default learning rate and epoch
# learner.autofit(0.005, 1)
# learner.fit_onecycle(lr = 2e-5, epochs = 1)

### Saving Model

predictor = ktrain.get_predictor(learner.model, preproc)
# predictor = ktrain.load_predictor('bert-ch-marc')

# predictor.save('../output/bert-ch-marc')

### Prediction

- Invoke `view_top_losses` to view the most misclassified review in the validation set

learner.view_top_losses(n=2, preproc=preproc)

import numpy as np
X_test_reviews =data_test['Reviews'].values
X_test_reviews[0]

X_test_len = [len(r) for r in X_test_reviews]
id = X_test_len.index(np.percentile(X_test_len, 90))
X_test_reviews[id]

print(predictor.predict(X_test_reviews[id]))
print(predictor.predict_proba(X_test_reviews[id]))
print(predictor.get_classes())

predictor.explain(X_test_reviews[id])

The visualization is generated using a technique called [LIME](https://arxiv.org/abs/1602.04938).  The input is randomly perturbed to examine how the prediction changes.  This is used to infer the relative importance of different words to the final prediction using a linear interpretable model.  

- The GREEN words contribute to the model prediction
- The RED (and PINK) words detract from the model prediction (Shade of color denotes the strength or size of the coefficients in the inferred linear model)

The model prediction is **positive**. Do GREEN words give the impression of a positive feedback?


## Logistic Regression

- Train a model to predict **Survival** using [Kaggle's Titatnic dataset](https://www.kaggle.com/c/titanic).

- After training the model, **explain** the model's prediction for a specific example.

## %cd '../../RepositoryData/data'

import pandas as pd
import numpy as np
train_df = pd.read_csv('titanic/train.csv', index_col=0)
train_df = train_df.drop('Name', 1)
train_df = train_df.drop('Ticket', 1)
train_df = train_df.drop('Cabin', 1)

np.random.seed(42)
p = 0.1 # 10% for test set
prop = 1-p
df = train_df.copy()
msk = np.random.rand(len(df)) < prop
train_df = df[msk]
test_df = df[~msk]


import ktrain
from ktrain import tabular
trn, val, preproc = tabular.tabular_from_df(train_df, label_columns=['Survived'], random_state=42)
model = tabular.tabular_classifier('mlp', trn) # multilayer perception (Deep Neural network)
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=32)
learner.fit_onecycle(1e-3, 25)

predictor = ktrain.get_predictor(learner.model, preproc)
preds = predictor.predict(test_df, return_proba=True)
df = test_df.copy()[[c for c in test_df.columns.values if c != 'Survived']]
df['Survived'] = test_df['Survived']
df['predicted_Survived'] = np.argmax(preds, axis=1)
df.head()

- Require [shap](https://github.com/slundberg/shap) library to perform model explanation
- take case ID35 for example


predictor.explain(test_df, row_index=35, class_id=1)

From the visualization above, we can see that:
- his First class status (`Pclass=1`) and his higher-than-average Fare price (suggesting that he is wealthy) are pushing the model higher towards predicting **Survived**. 
- the fact that he is a `Male` pushes the model to lower its prediction towards **NOT Survived**.
- For these reasons, this is a border-line and uncertain prediction.

## Regression

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="0"; 

import urllib.request
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)


import ktrain
from ktrain import tabular

## loading data
train_df = pd.read_csv('housing_price/train.csv', index_col=0)
train_df.head()

train_df.drop(['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'], 1, inplace=True)
train_df.head()


trn, val, preproc = tabular.tabular_from_df(train_df, is_regression=True, 
                                             label_columns='SalePrice', random_state=42)

model = tabular.tabular_regression_model('mlp', trn)
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=128)

learner.lr_find(show_plot=True, max_epochs=16)


## Inspect the loss plot above
learner.autofit(1e-1)

learner.evaluate(test_data=val)

predictor = ktrain.get_predictor(learner.model, preproc)

predictor.explain(train_df, row_index=25)

## References

- [*ktrain* Module](https://github.com/amaiya/ktrain)
- [Explainable AI in *ktrain*](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-A2-explaining-predictions.ipynb)
- [*ktrain* tutorial notebook on tabular models](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-08-tabular_classification_and_regression.ipynb)