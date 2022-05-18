#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis Using BERT

# - Using `ktrain` for learning
# - Using BERT pre-trained language model

# ## Installing ktrain

# In[1]:


get_ipython().system('pip install ktrain')


# ## Importing Libraries

# In[2]:


import tensorflow as tf
import pandas as pd
import numpy as np
import ktrain
from ktrain import text
import tensorflow as tf


# In[3]:


tf.__version__


# ## Clone Git Repository for Data

# In[4]:


get_ipython().system('git clone https://github.com/laxmimerit/IMDB-Movie-Reviews-Large-Dataset-50k.git')


# ## Data Preparation

# In[5]:


#loading the train dataset

data_train = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx', dtype = str)
#loading the test dataset

data_test = pd.read_excel('IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx', dtype = str)


# In[6]:


#dimension of the dataset

print("Size of train dataset: ",data_train.shape)
print("Size of test dataset: ",data_test.shape)
#printing last rows of train dataset

data_train.tail()
#printing head rows of test dataset

data_test.head()


# ## Train-Test Split

# In[7]:


# text.texts_from_df return two tuples
# maxlen means it is considering that much words and rest are getting trucated
# preprocess_mode means tokenizing, embedding and transformation of text corpus(here it is considering BERT model)


(X_train, y_train), (X_test, y_test), preproc = text.texts_from_df(train_df=data_train,
                                                                   text_column = 'Reviews',
                                                                   label_columns = 'Sentiment',
                                                                   val_df = data_test,
                                                                   maxlen = 500,
                                                                   preprocess_mode = 'bert')


# ## Define Model

# In[9]:


# name = "bert" means, here we are using BERT model.

model = text.text_classifier(name = 'bert',
                             train_data = (X_train, y_train),
                             preproc = preproc)


# ## Define Learner

# In[10]:


#here we have taken batch size as 6 as from the documentation it is recommend to use this with maxlen as 500

learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),
                   val_data = (X_test, y_test),
                   batch_size = 6)


# ## Fit Model

# In[10]:


#Essentially fit is a very basic training loop, whereas fit one cycle uses the one cycle policy callback

learner.fit_onecycle(lr = 2e-5, epochs = 1)


# ## Save Model

# In[11]:


predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('/content/drive/My Drive/ColabData/bert')


# ## Prediction

# In[13]:


#sample dataset to test on

data = ['this movie was horrible, the plot was really boring. acting was okay',
        'the fild is really sucked. there is not plot and acting was bad',
        'what a beautiful movie. great plot. acting was good. will see it again']


# In[14]:


predictor.predict(data)


# In[15]:


#return_proba = True means it will give the prediction probabilty for each class

predictor.predict(data, return_proba=True)


# In[16]:


#classes available

predictor.get_classes()


# In[17]:


# !zip -r /content/bert.zip /content/bert


# ## Deploy Model

# In[13]:


# #loading the model

predictor_load = ktrain.load_predictor('/content/drive/My Drive/ColabData/bert')


# In[19]:


# #predicting the data

# predictor_load.predict(data)


# ## References
# 
# - [`ktrain` module](https://github.com/amaiya/ktrain)
# - [Sentiment Classification Using Bert](https://kgptalkie.com/sentiment-classification-using-bert/)
# - [當Bert遇上Keras：這可能是Bert最簡單的打開姿勢](http://www.ipshop.xyz/15376.html)
# - [進擊的 BERT：NLP 界的巨人之力與遷移學習](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)
