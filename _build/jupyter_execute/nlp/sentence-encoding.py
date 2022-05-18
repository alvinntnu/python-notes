#!/usr/bin/env python
# coding: utf-8

# # Universal Sentence Embeddings

# - This is based on Ch 10 of Text Analytics with Python by Dipanjan Sarkar

# ## Loading Libaries

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd


# In[2]:


## Check GPU if any
# tf.test.is_gpu_available()
tf.test.gpu_device_name()
tf.config.list_physical_devices('GPU')


# ## Data
# 
# - The original [IMDB Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
#     - The original data include each text as an independent text file
# - [Sarkar's repository](https://github.com/dipanjanS/data_science_for_all/tree/master/tds_deep_transfer_learning_nlp_classification) for csv file

# In[3]:


# import tarfile
# tar = tarfile.open("../data/movie_review.tar.gz")
# tar.extractall(path="../data/stanford-movie-review/")
# tar.close()


# In[4]:


# import os
# import tarfile

# def csv_files(members):
#     for tarinfo in members:
#         if os.path.splitext(tarinfo.name)[1] == ".csv":
#             yield tarinfo

# tar = tarfile.open("../data/movie_review.tar.gz")
# tar.extractall(path='../data/', members=csv_files(tar))
# tar.close()


# In[5]:


dataset = pd.read_csv('../data/data_science_for_all-master/tds_deep_transfer_learning_nlp_classification/movie_reviews.csv.bz2',
                     compression='bz2')
dataset.info()


# In[6]:


dataset.dtypes


# In[7]:


## Recode sentiment

dataset['sentiment'] = [1 if sentiment=='positive' else 0 for sentiment in dataset['sentiment'].values]
dataset.head()


# In[8]:


dataset.dtypes


# ## Train, Validation, and Test Sets Splitting

# In[9]:


## Method 1 sklearn
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(reviews, test_size = 0.33, random_state=42)

## Method 2 numpy
train, validate, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.7*len(dataset))])


# In[10]:


train.shape, validate.shape, test.shape
train.head()


# ## Text Wranlging
# 
# - Text preprocessing usually takes care of:
#     - unnecessary html tags
#     - non-ASCII characters in English texts (e.g., accented characters)
#     - contraction issues
#     - special characters (unicode)

# In[11]:


## libaries for text pre-processing
## !pip3 install contractions
import contractions
from bs4 import BeautifulSoup
import unicodedata
import re


# In[19]:


## Functions for Text Preprocessing

def strip_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    [s.extract() for s in soup(['iframe','script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore'). decode('utf-8', 'ignore')
    return text
def expand_contractions(text): 
    return contractions.fix(text)
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]' 
    text = re.sub(pattern, '', text)
    return text
def pre_process_document(document):
    # strip HTML
    document = strip_html_tags(document)
    # case normalization
    document = document.lower()
    # remove empty lines
    document = document.translate(document.maketrans("\n\t\r", "   "))
    # remove accented characters
    document = remove_accented_chars(document)
    # expand contractions
    document = expand_contractions(document)
    # remove special characters and\or digits
    # insert spaces between special characters to isolate them 
    special_char_pattern = re.compile(r'([{.(-)!}])')
    document = special_char_pattern.sub(" \\1 ", document)
    document = remove_special_characters(document, remove_digits=True)
    # remove extra whitespace
    document = re.sub(' +', ' ', document) 
    document = document.strip()
    return document

# vectorize function
pre_process_corpus = np.vectorize(pre_process_document)


# In[20]:


pre_process_corpus(train['review'].values[0])


# In[21]:


get_ipython().run_cell_magic('time', '', "train_reviews = pre_process_corpus(train['review'].values)\ntrain_sentiments = train['sentiment'].values\nval_reviews = pre_process_corpus(validate['review'].values)\nval_sentiments = validate['sentiment'].values\ntest_reviews = pre_process_corpus(test['review'].values)\ntest_sentiments = test['sentiment'].values\n")


# ## Data Ingestion Functions for tensorflow

# In[28]:


# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    {'sentence': train_reviews}, train_sentiments, 
    batch_size=256, num_epochs=None, shuffle=True)


# In[29]:


# Prediction on the whole training set. 
predict_train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    {'sentence': train_reviews}, train_sentiments, shuffle=False)


# In[31]:


# Prediction on the whole validation set. 
predict_val_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    {'sentence': val_reviews}, val_sentiments, shuffle=False)


# In[32]:


# Prediction on the test set.
predict_test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    {'sentence': test_reviews}, test_sentiments, shuffle=False)


# ## Universal Sentence Encoder

# In[33]:


embedding_feature = hub.text_embedding_column(
    key='sentence', 
    module_spec="https://tfhub.dev/google/universal-sentence-encoder/2",
    trainable=False)


# In[37]:


dnn = tf.estimator.DNNClassifier(
    hidden_units=[512,128],
    feature_columns=[embedding_feature],
    n_classes=2,
    activation_fn=tf.nn.relu,
    dropout=0.1,
    optimizer=tf.optimizers.Adagrad(learning_rate=0.005))


# In[ ]:


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time

TOTAL_STEPS = 1500
STEP_SIZE = 100
for step in range(0, TOTAL_STEPS+1, STEP_SIZE):
    print()
    print('-'*100)
    print('Training for step =', step)
    start_time = time.time()
    dnn.train(input_fn=train_input_fn, steps=STEP_SIZE)
    elapsed_time = time.time() - start_time
    print('Train Time (s):', elapsed_time)
    print('Eval Metrics (Train):', dnn.evaluate(input_fn=predict_train_input_fn))
    print('Eval Metrics (Validation):', dnn.evaluate(input_fn=predict_val_input_fn))


# ## Model Evaluation

# In[ ]:


dnn.evaluate(input_fn=predict_train_input_fn)


# In[ ]:


dnn.evaluate(input_fn=predict_test_input_fn)

