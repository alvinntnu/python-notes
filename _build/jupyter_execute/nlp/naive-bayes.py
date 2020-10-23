# Naive Bayes

- Naive Bayes is a common traditional machine learning algorithm for classification task.
- Important assumptions behind Naive Bayes:
    - Features are independent of each other
    - Features have equal contributions to the prediction
- When applying Naive Bayes to text data, we need to convert text data into numeric features.
    - bag-of-words model
    - vectorization issues
    - Fine tune the vectorizing features for better representation of the texts
- Applications
    - Text Classification
- Issues
    - Fail to consider the sequential orders of words in texts

## Loading Libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

## Fetech Data

data = fetch_20newsgroups()

print(type(data))
print(len(data.filenames)) # doc num

text_categories = data.target_names
print(len(text_categories)) # total number of text categories

## Train-Test Split

train_data = fetch_20newsgroups(subset="train", categories = text_categories)
test_data = fetch_20newsgroups(subset="test", categories = text_categories)

print("There are {} unique classes (text categories)".format(len(text_categories)))
print("Training Sample Size: {}".format(len(train_data.data)))
print("Test Sample Size: {}".format(len(test_data.data)))

## Data Inspection

print(train_data.data[5][:200])

## Building Pipeline

- The modeling pipeline should include:
    - text transformation (vectorization)
    - naive bayes modeling

# Build model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Train the model using training data
model.fit(train_data.data, train_data.target)
# Predict the classes of test data
predicted_categories = model.predict(test_data.data)

## Evaluation

```{note}
By default, the confusion matrix indicate the correct labels on the rows and predicted labels on the columns.
```

accuracy_score(test_data.target, predicted_categories)

mat = confusion_matrix(test_data.target, predicted_categories)

import matplotlib
matplotlib.rcParams['figure.dpi']= 150

sns.heatmap(mat.T, square=True, annot=True, fmt="d",
           xticklabels=test_data.target_names,
           yticklabels=test_data.target_names,
           annot_kws={"size":4})
plt.xlabel("true labels")
plt.ylabel("predicted labels")
plt.show()

## References

- [Text Classification Using Naive Bayes: Theory & A Working Example](https://towardsdatascience.com/text-classification-using-naive-bayes-theory-a-working-example-2ef4b7eb7d5a)