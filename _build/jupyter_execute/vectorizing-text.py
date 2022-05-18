# Vectorizing Texts

- Feature engineering for text represenation

import nltk
from nltk.corpus import brown

brown.categories()

corpus_id = brown.fileids(categories=['reviews','fiction','humor'])
corpus_text = [' '.join(w) for w in [brown.words(fileids=cid) for cid in corpus_id]]
print(corpus_text[0][:100])
corpus_cat = [brown.categories(fileids=cid)[0] for cid in corpus_id]

print(len(corpus_text))
print(len(corpus_id))
print(len(corpus_cat))
type(corpus_id)

import numpy as np
import pandas as pd
import re

assert len(corpus_text)==len(corpus_cat)

corpus_df = pd.DataFrame({'Text': corpus_text, 'Category': corpus_cat, 'ID': corpus_id})
corpus_df

## Clean up texts
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)

## function
def normalize_text(text):
    ## remove special characters
    text = re.sub(r'[^a-zA-Z\s]','', text, re.I|re.A)
    text = text.lower().strip()
    tokens = wpt.tokenize(text)
    ## filtering
    #tokens_filtered = [w for w in tokens if w not in stop_words and re.search(r'\D+', w)]
    tokens_filtered = tokens
    text_output = ' '.join(tokens_filtered)
    
    return text_output

## vectorize function
normalize_corpus= np.vectorize(normalize_text)

corpus_norm = normalize_corpus(corpus_text)
corpus_norm[1][:200]

## Bag of Words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0.2, max_df=1.)
cv_matrix = cv.fit_transform(corpus_norm)
cv_matrix


## view the array
type(cv_matrix)

cv_matrix = cv_matrix.toarray()

vocab = cv.get_feature_names()
boa_unigram = pd.DataFrame(cv_matrix, columns=vocab)
boa_unigram

## More Complex Bag-of-Words

- Filter features based on word classes
- Include n-gram features

## Bag of N-grams Model

## N-grams

cv_ngram = CountVectorizer(ngram_range=(1,3), min_df = 0.2)
cv_ngram_matrix = cv_ngram.fit_transform(corpus_norm)

boa_ngram = pd.DataFrame(cv_ngram_matrix.toarray(),columns = cv_ngram.get_feature_names())
boa_ngram.head()

## TF-IDF Model

from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer(norm = 'l2', use_idf=True)
tf_matrix = tf.fit_transform(cv_matrix)

tfidf_unigram = pd.DataFrame(tf_matrix.toarray(), columns = cv.get_feature_names())
tfidf_unigram

tf_matrix2 = tf.fit_transform(cv_ngram_matrix)
tfidf_ngram = pd.DataFrame(tf_matrix2.toarray(), columns = cv_ngram.get_feature_names())
tfidf_ngram

- We can also create TF-IDF model directly from corpus

from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(min_df = 0.2,
                     ngram_range=(1,3), 
                     use_idf=True
                    )
tv_matrix = tv.fit_transform(corpus_norm)
tfidf_ngram2 = pd.DataFrame(tv_matrix.toarray(), columns=tv.get_feature_names())
tfidf_ngram2

## Document Similarity

- Cluster analysis with R seems more intuitive to me

from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(tv_matrix)
similarity_df = pd.DataFrame(similarity_matrix)
similarity_df

%load_ext rpy2.ipython

%%R -i similarity_matrix
library(dplyr)
#library(ggplot2)

head(similarity_matrix)
hclust(as.dist(similarity_matrix),method="ward.D2") %>%
plot

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(similarity_matrix, 'ward')
pd.DataFrame(Z)

# but to draw dendrogram?
dendrogram(Z)
# Don't like the graph in python