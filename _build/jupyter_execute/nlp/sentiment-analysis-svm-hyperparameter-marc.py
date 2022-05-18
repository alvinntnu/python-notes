#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Yahoo! Movie reviews
# 
# - Using Marc's thesis dataset (Chinese Yahoo Movie Reviews)
# - Important Steps
#   - Loading the CSV dataset
#   - Split train-test
#   - Define Pipeline classifier
#   - Find optimal hyper-parameters for SVM via Cross validation
#   - Model evaluation
#   - Model interpretation

# ## Setting Colab Environment
# 
# - Install a package `lime` for model interpretation

# In[1]:


get_ipython().system('pip install lime')


# ## Google Drive Access
# 
# - After running the code cell, visit the URL and copy-paste the code back here

# In[2]:


from google.colab import drive
drive.mount('/content/gdrive/')


# ## Loading Libraries

# In[18]:


get_ipython().system('pip install lime')
import pandas as pd
import numpy as np
import keras
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.pipeline import make_pipeline, TransformerMixin, Pipeline
from sklearn.base import BaseEstimator
from sklearn import metrics, svm
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from collections import OrderedDict
from lime.lime_text import LimeTextExplainer


# ## Loading Dataset

# In[4]:


df = pd.read_csv('/content/gdrive/My Drive/ColabData/marc_movie_review_metadata.csv')


# In[5]:


df.head()


# ## Train-Test Split

# In[6]:


## train-test split
reviews = df['reviews_sentiword_seg'].values
sentiments = df['rating'].values

X_train, X_test, y_train, y_test = train_test_split(  
    reviews, sentiments, test_size=0.1, random_state=0
)

print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)


# In[7]:


# import nltk
# from nltk.tokenize import WhitespaceTokenizer

# ## Transformer: Tokeniziation

# class ChineseTokenizer(WhitespaceTokenizer, BaseEstimator, TransformerMixin):
#     """ Sklearn transformer to convert texts to indices list 
#     (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""
#     def __init__(self,  **kwargs):
#         super().__init__(**kwargs)
        
#     def fit(self, y=None):
#         return self
    
#     def transform(self, texts, y=None):
#         return np.array(self.tokenize(texts))

# token_transform = ChineseTokenizer()


# ## Transformers and Pipeline

# In[8]:


## Transformer: BOW

ngram_range = (1,2)
min_df = 5

BOW_transform = CountVectorizer()


# In[9]:


# Estimator
classifier_svm = svm.SVC(C=1, kernel='linear', probability=True)



# In[10]:


## Create Pipeline

## To ensure the parameters are passed to the right transformer
## Make each step with an name
## Use the name in parameter setting of the GridSearch
## See below
pipeline = Pipeline([
  ('vectorizer',CountVectorizer(ngram_range=(1,1),min_df=5)), 
  ('clf', svm.SVC(C=1, kernel='linear'))])


# ## GridSearch Cross Validation
# 
# The hyper-parameters investigated here include:
# 
# - For SVM Classifier:
#   - C
#   - kernel
# - For Bag-of-Words Vectorizer:
#   - min_df
#   - ngram_range

# In[11]:


get_ipython().run_cell_magic('time', '', "## Hyper-parameter Tunning\nparameters = {\n    'clf__kernel':('linear','rbf'),\n    'clf__C':[1,10],\n    'vectorizer__ngram_range':[(1,1),(1,2), (1,3)],\n    'vectorizer__min_df':[2, 5,10]\n}\n\ncls = GridSearchCV(estimator=pipeline, param_grid = parameters)\ncls_cv_results = cls.fit(X_train, y_train)\n")


# ## Best Model and Model Prediction

# In[12]:


## Find the best model from cross-validation

print("Best: %f using %s" % (cls_cv_results.best_score_, cls_cv_results.best_params_))


# In[13]:


## Cross validation Results
import pandas as pd
cv_results_df=pd.DataFrame(cls.cv_results_)
cv_results_df


# In[14]:


## Prediction based on best model
y_preds = cross_val_predict(cls.best_estimator_, X_test, y_test)

## Model Testing
# from sklearn import metrics
# print('Computing predictions on test set...')
# y_preds = pipeline.predict(X_test)

print('Test accuracy: {:.2f} %'.format(100*metrics.accuracy_score(y_preds, y_test)))


# ## Interpretation
# 
# - Using LIME to interpret the importance of the features in relatio to the model prediction
# - Identify important words that may have great contribution to the model prediction
# - Based on [LIME of words: interpreting Recurrent Neural Networks predictions](https://data4thought.com/deep-lime.html)

# In[15]:


## Refit model based on optimal parameter settings
pipeline = Pipeline([
  ('vectorizer',CountVectorizer(ngram_range=(1,3),min_df=2)), 
  ('clf', svm.SVC(C=1, kernel='rbf', probability=True))])
pipeline.fit(X_train, y_train)


# In[19]:


import textwrap
reviews_test = X_test
sentiments_test = y_test



# We choose a sample from test set
idx = 45
text_sample = reviews_test[idx]
class_names = ['negative', 'positive']

print('Review ID-{}:'.format(idx))
print('-'*50)
print('Review Text:\n', textwrap.fill(text_sample,40))
print('-'*50)
print('Probability(positive) =', pipeline.predict_proba([text_sample])[0,1])
print('Probability(negative) =', pipeline.predict_proba([text_sample])[0,0])
print('Predicted class: %s' % pipeline.predict([text_sample]))
print('True class: %s' % sentiments_test[idx])


# In[25]:


#import matplotlib as plt
matplotlib.rcParams['figure.dpi']=300
get_ipython().run_line_magic('matplotlib', 'inline')


explainer = LimeTextExplainer(class_names=class_names)
explanation = explainer.explain_instance(text_sample, 
                                         pipeline.predict_proba, 
                                         num_features=20)
explanation.show_in_notebook(text=True)


# In[26]:


weights = OrderedDict(explanation.as_list())

print(weights)
lime_weights = pd.DataFrame({'words': list(weights.keys()), 'weights': list(weights.values())})

## Chinese fonts


def getChineseFont(size=15):  
    return matplotlib.font_manager.FontProperties(
        fname='/content/gdrive/My Drive/ColabData/SourceHanSans.ttc',size=size)  

print('Chinese Font Name:', getChineseFont().get_name())
sns.set(font=getChineseFont().get_name())
#sns.set_style('whitegrid',{'font.sans-serif':["Source Han Sans"]})
plt.figure(figsize=(5, 7), dpi=150)

sns.barplot(x="weights", y="words", data=lime_weights);
#sns.despine(left=True, bottom=True)
plt.yticks(rotation=0, FontProperties=getChineseFont(8))
plt.title('Review ID-{}: features weights given by LIME'.format(idx));


# In[27]:


# !wget -O taipei_sans_tc_beta.ttf https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download
# !mv taipei_sans_tc_beta.ttf /usr/local/lib/python3.6/dist-packages/matplotlib//mpl-data/fonts/ttf

