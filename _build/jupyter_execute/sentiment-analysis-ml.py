# Sentiment Analysis with Traditional Machine Learning

- This note is based on Text Analytics with Python Ch9 Senitment Analysis by Dipanjan Sarkar
- Logistic Regression
- Support Vector Machine (SVM)

## Import necessary depencencies

import pandas as pd
import numpy as np
#import text_normalizer as tn
#import model_evaluation_utils as meu
import nltk

np.set_printoptions(precision=2, linewidth=80)

## Load and normalize data

%%time
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

reviews[0][:100]
sentiments[0:10]
train_reviews[0][:100]
test_reviews[0][:100]
test_sentiments[0]

## Normalizing the Corpus

# normalize datasets
# stop_words = nltk.corpus.stopwords.words('english')
# stop_words.remove('no')
# stop_words.remove('but')
# stop_words.remove('not')

# norm_train_reviews = tn.normalize_corpus(train_reviews, stopwords=stop_words)
# norm_test_reviews = tn.normalize_corpus(test_reviews, stopwords=stop_words)

norm_train_reviews = train_reviews.tolist()
norm_test_reviews = test_reviews.tolist()

## Traditional Supervised Machine Learning Models

- Logistic
- SVM

## Feature Engineering

%%time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# build BOW features on train reviews
cv = CountVectorizer(binary=False, min_df=10, max_df=0.7, ngram_range=(1,3))
cv_train_features = cv.fit_transform(norm_train_reviews)
# build TFIDF features on train reviews
tv = TfidfVectorizer(use_idf=True, min_df=10, max_df=0.7, ngram_range=(1,3),
                     sublinear_tf=True)
tv_train_features = tv.fit_transform(norm_train_reviews)

# transform test reviews into features
cv_test_features = cv.transform(norm_test_reviews)
tv_test_features = tv.transform(norm_test_reviews)

print('BOW model:> Train features shape:', cv_train_features.shape, ' Test features shape:', cv_test_features.shape)
print('TFIDF model:> Train features shape:', tv_train_features.shape, ' Test features shape:', tv_test_features.shape)

## Model Training, Prediction and Performance Evaluation

from sklearn.linear_model import SGDClassifier, LogisticRegression

lr = LogisticRegression(penalty='l2', max_iter=200, C=1)
svm = SGDClassifier(loss='hinge', max_iter=200)

:::{note}
`pd.MultiIndex()` has been updated in Sarker's code. The argument `codes=` is new.
:::

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

%%time
# build model    
lr.fit(cv_train_features, train_sentiments)
# predict using model
lr_bow_predictions = lr.predict(cv_test_features) 

    
svm.fit(cv_train_features, train_sentiments)
svm_bow_predictions = svm.predict(cv_test_features)
    
# Logistic Regression model on BOW features
# lr_bow_predictions = meu.train_predict_model(classifier=lr, 
#                                              train_features=cv_train_features, train_labels=train_sentiments,
#                                              test_features=cv_test_features, test_labels=test_sentiments)

display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=lr_bow_predictions,
                                      classes=['positive','negative'])

display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=svm_bow_predictions,
                                      classes=['positive','negative'])

from sklearn.metrics import confusion_matrix
lr_bow_cm = confusion_matrix(test_sentiments, lr_bow_predictions)
svm_bow_cm = confusion_matrix(test_sentiments, svm_bow_predictions)
# lr_bow_cm.shape[1]
print(lr_bow_cm)
print(svm_bow_cm)


## MultiIndex DataFrame demo
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
# convert array to data frame
classes = ['positive','negative']
lr_bow_df_cm = pd.DataFrame(lr_bow_cm, 
                            index = pd.MultiIndex(levels=[['Actual'],classes],
                                                 codes=[[0,0],[0,1]]),
                            columns = pd.MultiIndex(levels=[['Predicted'],classes],
                                                 codes=[[0,0],[0,1]]))
lr_bow_df_cm

# pd.MultiIndex(levels=[['Predicted:'],['positive', 'negative']],
#              codes=[[0,0],[1,0]])

# classes=['Positive','Negative']
# total_classes = len(classes)
# level_labels = [total_classes*[0], list(range(total_classes))]
# print(total_classes)
# print(level_labels)

svm_bow_df_cm = pd.DataFrame(svm_bow_cm, index = ['positive', 'negative'],
                  columns = ['positive', 'negative'])
svm_bow_df_cm

plt.figure(figsize = (10,7))
sn.heatmap(lr_bow_df_cm, annot=True, fmt='.5g')

plt.figure(figsize = (10,7))
sn.heatmap(svm_bow_df_cm, annot=True, fmt='.5g')

display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=lr_bow_predictions,classes=['positive', 'negative'])

display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=svm_bow_predictions,classes=['positive', 'negative'])
    

# Logistic Regression model on TF-IDF features
# lr_tfidf_predictions = meu.train_predict_model(classifier=lr, 
#                                                train_features=tv_train_features, train_labels=train_sentiments,
#                                                test_features=tv_test_features, test_labels=test_sentiments)
#meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=lr_tfidf_predictions,
#                                      classes=['positive', 'negative'])

# svm_bow_predictions = meu.train_predict_model(classifier=svm, 
#                                              train_features=cv_train_features, train_labels=train_sentiments,
#                                              test_features=cv_test_features, test_labels=test_sentiments)
# meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=svm_bow_predictions,
#                                       classes=['positive', 'negative'])

# svm_tfidf_predictions = meu.train_predict_model(classifier=svm, 
#                                                 train_features=tv_train_features, train_labels=train_sentiments,
#                                                 test_features=tv_test_features, test_labels=test_sentiments)
# # meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=svm_tfidf_predictions,
#                                       classes=['positive', 'negative'])

## Explaining Model (LIME)

- See [LIME Documentationb](https://github.com/marcotcr/lime)

from lime import lime_text
from sklearn.pipeline import make_pipeline


c = make_pipeline(cv, lr)
print(c.predict_proba([norm_test_reviews[0]]))

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=['positive','negative'])

idx = 200
exp = explainer.explain_instance(norm_test_reviews[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(negative) =', c.predict_proba([norm_test_reviews[idx]])[0,1])

print('True class: %s' % test_sentiments[idx])

exp.as_list()

print('Original prediction:', lr.predict_proba(cv_test_features[idx])[0,1])
tmp = cv_test_features[idx].copy()
tmp[0,cv.vocabulary_['excellent']] = 0
tmp[0,cv.vocabulary_['see']] = 0
print('Prediction removing some features:', lr.predict_proba(tmp)[0,1])
print('Difference:', lr.predict_proba(tmp)[0,1] - lr.predict_proba(cv_test_features[idx])[0,1])

fig = exp.as_pyplot_figure()

exp.show_in_notebook(text=True)

## SVM

from sklearn.calibration import CalibratedClassifierCV 
calibrator = CalibratedClassifierCV(svm, cv='prefit')
svm2=calibrator.fit(cv_train_features, train_sentiments)

c2 = make_pipeline(cv, svm2)
print(c2.predict_proba([norm_test_reviews[0]]))



idx = 200
exp = explainer.explain_instance(norm_test_reviews[idx], c2.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(negative) =', c2.predict_proba([norm_test_reviews[idx]])[0,1])

print('True class: %s' % test_sentiments[idx])

exp.as_list()

print('Original prediction:', svm2.predict_proba(cv_test_features[idx])[0,1])
tmp = cv_test_features[idx].copy()
tmp[0,cv.vocabulary_['excellent']] = 0
tmp[0,cv.vocabulary_['well']] = 0
print('Prediction removing some features:', svm2.predict_proba(tmp)[0,1])
print('Difference:', svm2.predict_proba(tmp)[0,1] - lr.predict_proba(cv_test_features[idx])[0,1])

fig = exp.as_pyplot_figure()

exp.show_in_notebook(text=True)