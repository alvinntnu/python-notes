# Corpus Lingustics Methods

- With `nltk`, we can easily implement quite a few corpus-linguistic methods
    - Concordance Analysis (Simple Words)
    - Frequency Lists
    - Collocations
    - Data Analysis with R
    - Concordance Analysis (Patterns, Constructions?)
        - Patterns on sentence strings
        - Patterns on sentence word-tag strings

## Preparing Corpus Data

import nltk
from nltk.corpus import brown
from nltk.text import Text
import pandas as pd

brown_text = Text(brown.words())

## Collocations

- Documentation [nltk.collocations](https://www.nltk.org/howto/collocations.html)

## Collocations based on Text
brown_text.collocation_list()[:10]
#brown_text.collocations()

from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(brown.words())

## bigram collocations based on different association measures
finder.nbest(bigram_measures.likelihood_ratio,10)
finder.nbest(bigram_measures.pmi, 10)

## Apply freq-based filers for bigram collocations
finder.apply_freq_filter(10)

## Apply word filer
from nltk.corpus import stopwords
stop_words_en = stopwords.words('english')
finder.apply_word_filter(lambda x: not x.isalpha())

finder.nbest(bigram_measures.likelihood_ratio, 10)
finder.nbest(bigram_measures.pmi, 10)

## Create collocations based on tagged words
finder = BigramCollocationFinder.from_words(
    brown.tagged_words())
finder.apply_word_filter(lambda x: not x[0].isalpha())
finder.nbest(bigram_measures.pmi, 10)

## Create collcoations based on tags only
finder = BigramCollocationFinder.from_words(
    t for w, t in brown.tagged_words(tagset='universal'))
finder.nbest(bigram_measures.pmi, 10)

## Create collocations with intervneing words (gapped n-grams)
finder = BigramCollocationFinder.from_words(brown.words(), window_size=2)
finder.apply_word_filter(lambda x: not x.isalpha())
finder.apply_freq_filter(10)
finder.nbest(bigram_measures.pmi, 10)

## Finders
scored = finder.score_ngrams(bigram_measures.raw_freq)
scored[:10]

```{note}
How to get the document frequency of the bigrams???
```

unigram_freq = nltk.FreqDist(brown.words())
bigram_freq = nltk.FreqDist('_'.join(x) for x in nltk.bigrams(brown.words()))

unigram_freq_per_file = [nltk.FreqDist(words) 
                         for words in [brown.words(fileids=f) for f in brown.fileids()]]
bigram_freq_per_file = [nltk.FreqDist('_'.join(x) for x in nltk.bigrams(words))
                         for words in [brown.words(fileids=f) for f in brown.fileids()]]

## Function to get unigram dispersion
def createUnigramDipsersionDist(uni_freq, uni_freq_per_file):
    len(uni_freq_per_file)
    unigram_dispersion = {}

    for fid in uni_freq_per_file:
        for w, f in fid.items():
            if w in unigram_dispersion:
                unigram_dispersion[w] += 1
            else:
                unigram_dispersion[w] = 1
    return(unigram_dispersion)


unigram_dispersion = createUnigramDipsersionDist(unigram_freq, unigram_freq_per_file)
# Dictionary cannot be sliced/subset
# Get the items() and convert to list for subsetting
list(unigram_dispersion.items())[:20]

#dict(sorted(bigram_freq.items()[:3]))
list(bigram_freq.items())[:20]

bigram_dispersion = createUnigramDipsersionDist(bigram_freq, bigram_freq_per_file)
list(bigram_dispersion.items())[:20]

type(unigram_freq)
type(unigram_dispersion)

## Concordance

## Simple Concordances
brown_text.concordance('American', width=79, lines = 5)

## Regular Expression Concordances
import re
sents = [' '.join(s) for s in brown.sents()]
regex_1 = r'(is|was) \w+ing'
targets = [sent for sent in sents[:100] if re.search(regex_1, sent)]
targets[0]
#if targets:
#    for match in targets:
#        print(match.strip())

## Frequency List

## word frequencies
brown_fd_words = nltk.FreqDist(brown.words())
brown_fd_words.most_common(10)

## nouns freq
brown_df_nouns = nltk.FreqDist([w.lower() for w,t in brown.tagged_words() 
                                 if any (noun_tag in t for noun_tag in ['NP','NN'])])
brown_df_nouns.most_common(10)

brown_df_nouns_df = pd.DataFrame(brown_df_nouns.items(), columns=['word','freq'])
brown_df_nouns_df

%load_ext rpy2.ipython

%%R -i brown_df_nouns_df

library(dplyr)
brown_df_nouns_df %>%
filter(freq > 100) %>%
arrange(word) %>% 
head(50)