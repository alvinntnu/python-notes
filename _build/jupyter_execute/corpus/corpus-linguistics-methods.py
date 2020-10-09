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

## Delta P

## Inherit BigramAssocMeasures
class AugmentedBigramAssocMeasures(BigramAssocMeasures):
    @classmethod
    def dp_fwd(self, *marginals):
        """Scores bigrams using DP forward
        This may be shown with respect to a contingency table::

                w1    ~w1
             ------ ------
         w2 | n_ii | n_oi | = n_xi
             ------ ------
        ~w2 | n_io | n_oo |
             ------ ------
             = n_ix        TOTAL = n_xx
        """
        
        n_ii, n_io, n_oi, n_oo = self._contingency(*marginals)

        return (n_ii/(n_ii+n_io)) - (n_oi/(n_oi+n_oo))

    @classmethod
    def dp_bwd(self, *marginals):
        """Scores bigrams using DP backward
        This may be shown with respect to a contingency table::

                w1    ~w1
             ------ ------
         w2 | n_ii | n_oi | = n_xi
             ------ ------
        ~w2 | n_io | n_oo |
             ------ ------
             = n_ix        TOTAL = n_xx
        """
        
        n_ii, n_io, n_oi, n_oo = self._contingency(*marginals)

        return (n_ii/(n_ii+n_oi)) - (n_io/(n_io+n_oo))

bigram_measures = AugmentedBigramAssocMeasures()
finder = BigramCollocationFinder.from_words(brown.words())

finder.apply_freq_filter(10)

bigrams_dpfwd = finder.score_ngrams(bigram_measures.dp_fwd)
bigrams_dpfwd[:10]

bigrams_dpbwd = finder.score_ngrams(bigram_measures.dp_bwd)
bigrams_dpbwd[:10]

## Concordance

## Simple Concordances
brown_text.concordance('American', width=79, lines = 5)

#nltk.app.concordance()

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
brown_fd_nouns = nltk.FreqDist([w.lower() for w,t in brown.tagged_words() 
                                 if any (noun_tag in t for noun_tag in ['NP','NN'])])
brown_fd_nouns.most_common(10)

brown_fd_nouns_df = pd.DataFrame(brown_fd_nouns.items(), columns=['word','freq'])

Sort the data frame:

brown_fd_nouns_df[brown_fd_nouns_df['freq']>100].sort_values(["freq","word"],ascending=[False,True])

```{note}
We can also pass the data frame to R for data exploration.
```

%load_ext rpy2.ipython

%%R -i brown_fd_nouns_df

library(dplyr)
brown_fd_nouns_df %>%
filter(freq > 100) %>%
arrange(desc(freq), word) %>% 
head(50)

## Conditional Frequency List


## Word by POS Frequency Distribution

brown_news_tagged_words = brown.tagged_words(categories='news', tagset='universal')
brown_news_cfd = nltk.ConditionalFreqDist(brown_news_tagged_words)
brown_news_cfd['yield']

## POS by Word Frequency Distribution
brown_news_cfd2 = nltk.ConditionalFreqDist([(t, w) for (w, t) in brown_news_tagged_words])
brown_news_cfd2['VERB'].most_common(10)

## Word by Genre Frequency Distribution
brown_genre_cfd = nltk.ConditionalFreqDist(
    (word, genre)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

brown_genre_cfd.conditions()[:50]
brown_genre_cfd['mysterious']

print(sorted(brown_genre_cfd['mysterious'].items(),key=lambda x:x[1],reverse=True)) # with freq

## Genre by Word Frequency Distribution
brown_genre_cdf2 = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

## Genre by Word Frequency Distribution
brown_genre_cdf2 = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)




top_n_word = [word for (word, freq) in brown_fd_words.most_common(20) if word[0].isalpha()]

brown_genre_cdf2.tabulate(conditions=['adventure','editorial','fiction'],
                         samples=top_n_word[:10])

top_n_word2 = [word for (word, tag) in brown.tagged_words(tagset='universal') 
               if tag.startswith('NOUN')]
top_n_word2_fd = nltk.FreqDist(top_n_word2).most_common(10)

print(top_n_word2_fd)
brown_genre_cdf2.tabulate(conditions=['adventure','editorial','fiction'],
                         samples=[w for (w, f) in top_n_word2_fd])