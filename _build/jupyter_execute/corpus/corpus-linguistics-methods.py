# Corpus Lingustics Methods

- With `nltk`, we can easily implement quite a few corpus-linguistic methods
    - Concordance Analysis (Simple Word Search)
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

## Dispersion

- Dispersion of a linguistic unit is also important.
- There should be a metric that indicates how evenly distributed the linguistic unit is.

```{note}
How to get the document frequency of the bigrams???
```

unigram_freq = nltk.FreqDist(brown.words())
bigram_freq = nltk.FreqDist('_'.join(x) for x in nltk.bigrams(brown.words()))

# ngram freq list of each file in the corpus
unigram_freq_per_file = [nltk.FreqDist(words) 
                         for words in [brown.words(fileids=f) for f in brown.fileids()]]
bigram_freq_per_file = [nltk.FreqDist('_'.join(x) for x in nltk.bigrams(words))
                         for words in [brown.words(fileids=f) for f in brown.fileids()]]

## Function to get unigram dispersion
def createDipsersionDist(uni_freq, uni_freq_per_file):
    len(uni_freq_per_file)
    unigram_dispersion = {}

    for fid in uni_freq_per_file:
        for w, f in fid.items():
            if w in unigram_dispersion:
                unigram_dispersion[w] += 1
            else:
                unigram_dispersion[w] = 1
    return(unigram_dispersion)


unigram_dispersion = createDipsersionDist(unigram_freq, unigram_freq_per_file)
# Dictionary cannot be sliced/subset
# Get the items() and convert to list for subsetting
list(unigram_dispersion.items())[:20]

#dict(sorted(bigram_freq.items()[:3]))
list(bigram_freq.items())[:20]

bigram_dispersion = createDipsersionDist(bigram_freq, bigram_freq_per_file)
list(bigram_dispersion.items())[:20]

type(unigram_freq)
type(unigram_dispersion)

:::{note}
We can implement the Delta P dispersion metric proposed by [Gries (2008)](https://www.researchgate.net/publication/233685362_Dispersions_and_adjusted_frequencies_in_corpora).
:::

## Delta P

- This is a directional association metric.

## Inherit BigramAssocMeasures
class AugmentedBigramAssocMeasures(BigramAssocMeasures):
    @classmethod
    def raw_freq2(cls,*marginals):          
        """Scores ngrams by their frequency"""
        n_ii, n_io, n_oi, n_oo = cls._contingency(*marginals)
        return n_ii
    
    @classmethod
    def dp_fwd(cls, *marginals):
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
        
        n_ii, n_oi, n_io, n_oo = cls._contingency(*marginals)

        return (n_ii/(n_ii+n_io)) - (n_oi/(n_oi+n_oo))

    @classmethod
    def dp_bwd(cls, *marginals):
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
        
        n_ii, n_oi, n_io, n_oo = cls._contingency(*marginals)

        return (n_ii/(n_ii+n_oi)) - (n_io/(n_io+n_oo))

bigram_measures = AugmentedBigramAssocMeasures()
finder = BigramCollocationFinder.from_words(brown.words())

#finder.apply_freq_filter(10)

bigrams_dpfwd = finder.score_ngrams(bigram_measures.dp_fwd)
bigrams_dpfwd[:10]

bigrams_dpbwd = finder.score_ngrams(bigram_measures.dp_bwd)
bigrams_dpbwd[:10]

## Checking Computation Accuracy

- Check if DP is correctly computed.

bigrams_rawfreq = finder.score_ngrams(bigram_measures.raw_freq2)

bigrams_rawfreq[:10]

unigrams_rawfreq = nltk.FreqDist(brown.words())

w1f = unigrams_rawfreq['of']
w2f = unigrams_rawfreq['the']
w1w2 = [freq for (w1,w2),freq in bigrams_rawfreq if w1=="of" and w2=="the"][0]
corpus_size = np.sum(list(unigrams_rawfreq.values()))

"""
        w1     _w1
w2      w1w2   ____    w2f
_w2     ____   ____
        w1f            corpus_size
"""

print(w1f, w2f, w1w2,corpus_size)

print('Delta P Forward for `of the`:', (w1w2/(w1f))-((w2f-w1w2)/(corpus_size-w1f)))
print('Delta P Backward for `of the`:', (w1w2/(w2f))-((w1f-w1w2)/(corpus_size-w2f)))

print([dp for (w1, w2),dp in bigrams_dpfwd if w1=="of" and w2=="the"])
print([dp for (w1, w2),dp in bigrams_dpbwd if w1=="of" and w2=="the"])

```{note}
How to implement the delta P of trigrams?
```

# inherit Trigram
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
class AugmentedTrigramAssocMeasures(TrigramAssocMeasures):
    """
    A collection of trigram association measures. Each association measure
        is provided as a function with four arguments::

            trigram_score_fn(n_iii,
                             (n_iix, n_ixi, n_xii),
                             (n_ixx, n_xix, n_xxi),
                             n_xxx)

        The arguments constitute the marginals of a contingency table, counting
        the occurrences of particular events in a corpus. The letter i in the
        suffix refers to the appearance of the word in question, while x indicates
        the appearance of any word. Thus, for example:
        n_iii counts (w1, w2, w3), i.e. the trigram being scored
        n_ixx counts (w1, *, *)
        n_xxx counts (*, *, *), i.e. any trigram
    """
    
    @classmethod
    def dp_fwd(cls, *marginals):
        """
        Scores trigrams using delta P forward, i.e. conditional prob of w3 given w1,w2
        minus conditional prob of w3, in the absence of w1,w2
        """
        n_iii, n_oii, n_ioi, n_ooi, n_iio, n_oio, n_ioo, n_ooo = cls._contingency(*marginals)

        return ((n_iii)/(n_iii+n_iio)) - ((n_ooi)/(n_ooi+n_ooo))
    @classmethod
    def dp_bwd(cls, *marginals):
        """
        Scores trigrams using delta P backward, i.e. conditional prob of w1 given w2,w3
        minus conditional prob of w1, in the absence of w2,w3
        """
        n_iii, n_oii, n_ioi, n_ooi, n_iio, n_oio, n_ioo, n_ooo = cls._contingency(*marginals)

        return ((n_iii)/(n_iii+n_oii)) - ((n_ioo)/(n_ioo+n_ooo))

trigram_measures = AugmentedTrigramAssocMeasures()
finder3 = TrigramCollocationFinder.from_words(brown.words())

finder3.apply_freq_filter(10)

finder3.nbest(trigram_measures.pmi, 10)

finder3.nbest(trigram_measures.dp_fwd, 10)

finder3.nbest(trigram_measures.dp_bwd,10)

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


```{toctree}
:hidden:
:titlesonly:


lexical-bundles
tokenization
word-cloud
patterns-constructions
```
