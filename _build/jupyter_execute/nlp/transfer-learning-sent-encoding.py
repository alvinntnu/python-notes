#!/usr/bin/env python
# coding: utf-8

# # Universal Sentence Embeddings

# - This is based on Ch 10 of Text Analytics with Python by Dipanjan Sarkar
# - [source](https://www.curiousily.com/posts/sentiment-analysis-with-tensorflow-2-and-keras-using-python/)
# - [source](https://www.dlology.com/blog/keras-meets-universal-sentence-encoder-transfer-learning-for-text-data/)

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

# In[114]:


# import tarfile
# tar = tarfile.open("../data/movie_review.tar.gz")
# tar.extractall(path="../data/stanford-movie-review/")
# tar.close()


# In[115]:


# import os
# import tarfile

# def csv_files(members):
#     for tarinfo in members:
#         if os.path.splitext(tarinfo.name)[1] == ".csv":
#             yield tarinfo

# tar = tarfile.open("../data/movie_review.tar.gz")
# tar.extractall(path='../data/', members=csv_files(tar))
# tar.close()


# In[116]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[117]:


dataset = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/movie_reviews.csv.bz2',
                     compression='bz2')
dataset.info()


# In[118]:


dataset.dtypes


# In[119]:


## Recode sentiment

dataset['sentiment'] = [1 if sentiment=='positive' else 0 for sentiment in dataset['sentiment'].values]
dataset.head()


# In[120]:


dataset.dtypes


# ## Train, Validation, and Test Sets Splitting

# In[10]:


## Method 1 sklearn
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(reviews, test_size = 0.33, random_state=42)

## Method 2 numpy
train, validate, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.7*len(dataset))])


# In[11]:


train.shape, validate.shape, test.shape
train.head()


# ## Text Wranlging
# 
# - Text preprocessing usually takes care of:
#     - unnecessary html tags
#     - non-ASCII characters in English texts (e.g., accented characters)
#     - contraction issues
#     - special characters (unicode)

# In[12]:


## libaries for text pre-processing
get_ipython().system('pip3 install contractions')
import contractions
from bs4 import BeautifulSoup
import unicodedata
import re


# In[13]:


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


# In[14]:


pre_process_corpus(train['review'].values[0])


# In[15]:


get_ipython().run_cell_magic('time', '', "train_reviews = pre_process_corpus(train['review'].values)\ntrain_sentiments = train['sentiment'].values\nval_reviews = pre_process_corpus(validate['review'].values)\nval_sentiments = validate['sentiment'].values\ntest_reviews = pre_process_corpus(test['review'].values)\ntest_sentiments = test['sentiment'].values\n")


# In[ ]:





# In[27]:


# train_text = train_reviews.tolist()
train_text = np.array(train_reviews, dtype=object)[:, np.newaxis]
test_text = np.array(test_reviews, dtype=object)[:, np.newaxis]
val_text = np.array(val_reviews, dtype=object)[:, np.newaxis]

train_label = np.asarray(pd.get_dummies(train_sentiments), dtype = np.int8)
test_label = np.asarray(pd.get_dummies(test_sentiments), dtype = np.int8)
val_label = np.asarray(pd.get_dummies(val_sentiments), dtype = np.int8)


# In[28]:


print(train_text[1])
print(train_label[1]) # y is one-hot encoding


# In[20]:


import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])



# In[ ]:


print(len(embeddings[0]))
print(embed(train_text[1])) # train_text[0] sentence embeddings


# In[25]:


from tqdm import tqdm

## Converting train_text into embedding
X_train = []
for r in tqdm(train_text):
  emb = embed(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_train.append(review_emb)
X_train = np.array(X_train)


# In[29]:


## Converting test_text into embeddings
X_test = []
for r in tqdm(test_text):
  emb = embed(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_test.append(review_emb)
X_test = np.array(X_test)


# In[31]:


## Converting val_text into embeddings
X_val = []
for r in tqdm(val_text):
  emb = embed(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_val.append(review_emb)
X_val = np.array(X_val)


# In[33]:


import keras
model = keras.Sequential()
model.add(
  keras.layers.Dense(
    units=256,
    input_shape=(X_train.shape[1], ),
    activation='relu'
  )
)
model.add(
  keras.layers.Dropout(rate=0.5)
)
model.add(
  keras.layers.Dense(
    units=128,
    activation='relu'
  )
)
model.add(
  keras.layers.Dropout(rate=0.5)
)
model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)


# In[146]:


history = model.fit(
    X_train, train_label,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
    shuffle=True
)


# In[147]:


## plotting 

import pandas as pd

history.history

history_df = pd.DataFrame(list(zip(history.history['loss'],history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy'])),
                          columns=['loss','accuracy','val_loss','val_accurary'])
history_df['epoch']=list(range(1,len(history_df['loss'])+1,1))


# In[52]:


history_df


# In[148]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel("epoch")
plt.ylabel("Cross-entropy loss")
plt.legend();


# # style
# plt.style.use('seaborn-darkgrid')
 
# # create a color palette
# palette = plt.get_cmap('Set1')
 
# # multiple line plot
# num=0
# for column in history_df.drop(['epoch','loss','val_loss'], axis=1):
#   num+=1
#   plt.plot(history_df['epoch'], history_df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
 
# # Add legend
# plt.legend(loc=2, ncol=2)
 
# # Add titles
# plt.title("A (bad) Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')
# plt.xlabel("Time")
# plt.ylabel("Score")


# In[71]:


plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend();


# In[149]:


model.evaluate(X_test, test_label)


# In[136]:


y_pred = model.predict(X_test) 
#y_pred = np.argmax(y_pred, axis = 1)[:5] 
label = np.argmax(test_label,axis = 1)[:5] 
print(y_pred[:5])
print(label[:5])


# In[128]:


y_pred = model.predict(X_test[:1])
print(y_pred)

"Bad" if np.argmax(y_pred) == 0 else "Good"


# In[131]:


print(test_text[1])


# In[150]:


# functions from Text Analytics with Python book
def get_metrics(true_labels, predicted_labels):
    
    print('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        4))
    print('Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))

def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]):
    
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, 
                                  labels=classes)
    cm_frame = pd.DataFrame(data=cm, 
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes], 
                                                  codes=level_labels), 
                            index=pd.MultiIndex(levels=[['Actual:'], classes], 
                                                codes=level_labels)) 
    print(cm_frame) 
def display_classification_report(true_labels, predicted_labels, classes=[1,0]):

    report = metrics.classification_report(y_true=true_labels, 
                                           y_pred=predicted_labels, 
                                           labels=classes) 
    print(report)
    
    
    
def display_model_performance_metrics(true_labels, predicted_labels, classes=[1,0]):
    print('Model Performance metrics:')
    print('-'*30)
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    print('\nModel Classification report:')
    print('-'*30)
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, 
                                  classes=classes)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, 
                             classes=classes)
from sklearn import metrics


# In[151]:


test_pred = model.predict(X_test)
test_pred = np.argmax(test_pred, axis=1)
print(test_pred)
print(test_sentiments)
display_model_performance_metrics(test_sentiments, test_pred)

