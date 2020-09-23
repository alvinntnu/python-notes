# Sentiment Analysis with Deep Learning

## Loading Packages

%%time
import gensim
import keras
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from sklearn.preprocessing import LabelEncoder
from keras.layers.normalization import BatchNormalization

## Preparing Data

## Data Import and Preprocessing
import pandas as pd
import numpy as np
#import text_normalizer as tn
#import model_evaluation_utils as meu
import nltk

np.set_printoptions(precision=2, linewidth=80)

dataset = pd.read_csv('../data/movie_reviews.csv')
# take a peek at the data
print(dataset.head())
reviews = np.array(dataset['review'])
sentiments = np.array(dataset['sentiment'])
type(reviews)
reviews.shape
sentiments.shape
# build train and test datasets
train_reviews = reviews[:35000]
train_sentiments = sentiments[:35000]
test_reviews = reviews[35000:]
test_sentiments = sentiments[35000:]

## Processing is ignored

norm_train_reviews = train_reviews
norm_test_reviews = test_reviews

from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()

le = LabelEncoder()
num_classes=2 
# tokenize train reviews & encode train labels
tokenized_train = [tokenizer.tokenize(text)
                   for text in norm_train_reviews]
y_tr = le.fit_transform(train_sentiments)
y_train = keras.utils.to_categorical(y_tr, num_classes)

# tokenize test reviews & encode test labels
tokenized_test = [tokenizer.tokenize(text)
                   for text in norm_test_reviews]
y_ts = le.fit_transform(test_sentiments)
y_test = keras.utils.to_categorical(y_ts, num_classes)



# print class label encoding map and encoded labels
print('Sentiment class label map:', dict(zip(le.classes_, le.transform(le.classes_))))
print('Sample test label transformation:\n'+'-'*35,
      '\nActual Labels:', test_sentiments[:3], '\nEncoded Labels:', y_ts[:3], 
      '\nOne hot encoded Labels:\n', y_test[:3])

## Training Word Embeddings

%%time
# build word2vec model
w2v_num_features = 512
w2v_model = gensim.models.Word2Vec(tokenized_train, size=w2v_num_features, window=150,
                                   min_count=10, sample=1e-3, workers=16)    

## takes 5mins

## This model uses the document word vector averaging scheme
## Use the average word vector representations to represent one document (movie reivew)

def averaged_word2vec_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    
    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,), dtype="float64")
        nwords = 0.
        
        for word in words:
            if word in vocabulary: 
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model.wv[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)

# generate averaged word vector features from word2vec model
avg_wv_train_features = averaged_word2vec_vectorizer(corpus=tokenized_train, model=w2v_model,
                                                     num_features=w2v_num_features)
avg_wv_test_features = averaged_word2vec_vectorizer(corpus=tokenized_test, model=w2v_model,
                                                    num_features=w2v_num_features)

## Loading Pre-trained Word Embeddings

# %%time
# # Use the 300-dimensional word vectors trained on the Common Crawl using the GloVe model
# # Provided by spaCy

# import spacy
# #nlp = spacy.load('en', parse=False, tag=False, entity=False)
# nlp_vec = spacy.load('en_vectors_web_lg', parse=False, tag=False, entity=False)

# ## feature engineering with GloVe model
# train_nlp = [nlp_vec(item) for item in norm_train_reviews]
# train_glove_features = np.array([item.vector for item in train_nlp])

# test_nlp = [nlp_vec(item) for item in norm_test_reviews]
# test_glove_features = np.array([item.vector for item in test_nlp])

# print('Word2Vec model:> Train features shape:', avg_wv_train_features.shape, ' Test features shape:', avg_wv_test_features.shape)
# print('GloVe model:> Train features shape:', train_glove_features.shape, ' Test features shape:', test_glove_features.shape)

## Building Model

- A simple fully-connected 4 layer deep neural network
    - input layer (not counted as one layer), i.e., the word embedding layer
    - three dense hidden layers (with 512 neurons)
    - one output layer (with 2 neurons for classification)
- (aka. multi-layered perceptron or deep ANN)

def construct_deepnn_architecture(num_input_features):
    dnn_model = Sequential()
    dnn_model.add(Dense(512, input_shape=(num_input_features,), kernel_initializer='glorot_uniform'))
    dnn_model.add(BatchNormalization()) # improve  stability of the network.
    dnn_model.add(Activation('relu')) # relu better than sigmoid, to present vanishing gradient problem
    dnn_model.add(Dropout(0.2)) # prevents overfitting
    
    dnn_model.add(Dense(512, kernel_initializer='glorot_uniform'))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Activation('relu'))
    dnn_model.add(Dropout(0.2))
    
    dnn_model.add(Dense(512, kernel_initializer='glorot_uniform'))
    dnn_model.add(BatchNormalization())
    dnn_model.add(Activation('relu'))
    dnn_model.add(Dropout(0.2))
    
    dnn_model.add(Dense(2))
    dnn_model.add(Activation('softmax'))

    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam',                 
                      metrics=['accuracy'])
    return dnn_model

w2v_dnn = construct_deepnn_architecture(num_input_features=w2v_num_features)

## Model Visualization

- To make this work, install `pip3 install pydot`
- and also install `!brew install graphviz` in terminal for mac
    - that is, install [graphvis](https://graphviz.gitlab.io/download/)


## Not working yet. Had a problem with the installation of graphviz on mac

# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# SVG(model_to_dot(w2v_dnn, show_shapes=True, show_layer_names=False, 
#                  rankdir='TB').create(prog='dot', format='svg'))

## Model Fitting

### Fitting using self-trained word embeddings

batch_size = 100
w2v_dnn.fit(avg_wv_train_features, y_train, epochs=10, batch_size=batch_size, 
            shuffle=True, validation_split=0.1, verbose=1)

y_pred = w2v_dnn.predict_classes(avg_wv_test_features)
predictions = le.inverse_transform(y_pred) 

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

display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=predictions, 
                                      classes=['positive', 'negative'])  

### Fitting using pre-trained word embedding model

# glove_dnn = construct_deepnn_architecture(num_input_features=300)

# batch_size = 100
# glove_dnn.fit(train_glove_features, y_train, epochs=10, batch_size=batch_size, 
#               shuffle=True, validation_split=0.1, verbose=1)

# y_pred = glove_dnn.predict_classes(test_glove_features)
# predictions = le.inverse_transform(y_pred) 

# meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=predictions, 
#                                       classes=['positive', 'negative'])  