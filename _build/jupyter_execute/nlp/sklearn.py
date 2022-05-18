# Machine Learning with Sci-Kit Learn

- Based on [Keith Galli's sklearn tutorial](https://github.com/KeithGalli/sklearn)

## Data Class

- Create a Review class for each token of the data
- This also demonstrates how Object-Oriented language works with the class

import random

class Sentiment:
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'
    POSITIVE = 'POSITIVE'

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
    def get_text(self):
        return [x.text for x in self.reviews]
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

## Data Loading

import json

fname = '../data/books_small_10000.json'

reviews = []
with open(fname) as f:
    for line in f:
        cur_review = json.loads(line)
        reviews.append(Review(cur_review['reviewText'], cur_review['overall']))

print('Number of Reviews: {}'.format(len(reviews)))
print('Sample Text of Doc 1:')
print('-'*30)
print(reviews[0].text)

## Check Sentiment Distribution of the Current Dataset
from collections import Counter
sentiment_distr = Counter([r.get_sentiment() for r in reviews])
print(sentiment_distr)

## Splitting Data into Train and Test Sets

from sklearn.model_selection import train_test_split
train, test = train_test_split(reviews, test_size = 0.33, random_state=42)

## Sentiment Distrubtion for Train and Test
print(Counter([r.get_sentiment() for r in train]))
print(Counter([r.get_sentiment() for r in test]))

## Balance the Classes

train_container = ReviewContainer(train)
test_container = ReviewContainer(test)

# balance
train_container.evenly_distribute()
test_container.evenly_distribute()

# check sentiment distribution again
print(Counter([r.get_sentiment() for r in train_container.reviews]))
print(Counter([r.get_sentiment() for r in test_container.reviews]))



## Train and Test Data and Labels

train_text = train_container.get_text()
train_label = train_container.get_sentiment()
test_text = test_container.get_text()
test_label = test_container.get_sentiment()

# print(train_label.count(Sentiment.POSITIVE))
# print(train_label.count(Sentiment.NEGATIVE))
print(Counter(train_label))

## Vectorization: Bag-of-Words Model

:::{admonition,important}

- Always split the data into train and test first before vectorizing the texts
- Otherwise, you would leak information to the training process, which may lead to over-fitting
- When vectorizing the texts, `fit_transform()` the train and `transform()` the test
:::

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tfidf_vec = TfidfVectorizer()
train_text_bow = tfidf_vec.fit_transform(train_text) # fit train
test_text_bow = tfidf_vec.transform(test_text) # transform test

print(train_text_bow.shape)
print(test_text_bow.shape)
print(type(train_text_bow))
print(train_text_bow[0])

## Classification Models

### Support Vector Machine (SVM)

from sklearn import svm

model_svm = svm.SVC(kernel='linear')
model_svm.fit(train_text_bow, train_label)

model_svm.predict(test_text_bow[:10])
#print(model_svm.score(test_text_bow, test_label))

### Decision Tree

from sklearn.tree import DecisionTreeClassifier

model_dec = DecisionTreeClassifier()
model_dec.fit(train_text_bow, train_label)

model_dec.predict(test_text_bow[:10])

### Naive Bayes

from sklearn.naive_bayes import GaussianNB
model_gnb = GaussianNB()
model_gnb.fit(train_text_bow.toarray(), train_label)

model_gnb.predict(test_text_bow[:10].toarray())

### Logistic Regression

from sklearn.linear_model import LogisticRegression

model_lg = LogisticRegression()
model_lg.fit(train_text_bow, train_label)

model_lg.predict(test_text_bow[:10].toarray())

## Evaluation

#Mean Accuracy
print(model_svm.score(test_text_bow, test_label))
print(model_dec.score(test_text_bow, test_label))
print(model_gnb.score(test_text_bow.toarray(), test_label))
print(model_lg.score(test_text_bow, test_label))

# F1
from sklearn.metrics import f1_score

f1_score(test_label, model_svm.predict(test_text_bow), average=None, labels = [Sentiment.POSITIVE, Sentiment.NEGATIVE])

## try a whole new self-created review:)
new_review =['This book looks soso like the content but the cover is weird',
             'This book looks soso like the content and the cover is weird'
            ]
new_review_bow = tfidf_vec.transform(new_review)
model_svm.predict(new_review_bow)

## Tuning Model

from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_text_bow, train_label)

sorted(clf.cv_results_.keys())
print(clf.best_params_)

print(clf.score(test_text_bow, test_label))

## Saving Model


#  import pickle

# with open('../ml-sent-svm.pkl', 'wb') as f:
#     pickle.dump(clf, f)
# with open('../ml-sent-svm.pkl' 'rb') as f:
#     loaded_svm = pickle.load(f)

# import pkg_resources
# import types
# def get_imports():
#     for name, val in globals().items():
#         if isinstance(val, types.ModuleType):
#             # Split ensures you get root package, 
#             # not just imported function
#             name = val.__name__.split(".")[0]

#         elif isinstance(val, type):
#             name = val.__module__.split(".")[0]

#         # Some packages are weird and have different
#         # imported names vs. system/pip names. Unfortunately,
#         # there is no systematic way to get pip names from
#         # a package's imported name. You'll have to add
#         # exceptions to this list manually!
#         poorly_named_packages = {
#             "PIL": "Pillow",
#             "sklearn": "scikit-learn"
#         }
#         if name in poorly_named_packages.keys():
#             name = poorly_named_packages[name]

#         yield name

# get_imports()

# imports = list(set(get_imports()))

# # The only way I found to get the version of the root package
# # from only the name of the package is to cross-check the names 
# # of installed packages vs. imported packages
# requirements = []
# for m in pkg_resources.working_set:
#     if m.project_name in imports and m.project_name!="pip":
#         requirements.append((m.project_name, m.version))

# for r in requirements:
#     print("{}=={}".format(*r))

