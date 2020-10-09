# Text Summarization

DEMO_DATA_ROOT = "../../../RepositoryData/data"

- Efficient ways to summarize the semantics of massive collections of documents
- Three general methods:
    - Keyphrase extraction
    - Topic modeling
    - Document summarization

## Keyphrase Extraction

- N-grams
- Collocations

from nltk.corpus import gutenberg
import text_normalizer as tn
import nltk
from operator import itemgetter

Loading corpus, Alice in the Wonderland.

# load corpus
alice = gutenberg.sents(fileids='carroll-alice.txt')
# concatenate each word token of a sentence
alice = [' '.join(ts) for ts in alice]
# normalize text
# `filter()` removes tokens that are False after normalization
norm_alice = list(filter(None, 
                         tn.normalize_corpus(alice, text_lemmatization=False))) 

Compare raw texts vs, noramlized texts:

print(alice[0], '\n', norm_alice[0])

### N-grams

A function to create n-grams.

def compute_ngrams(sequence, n):
    return list(
        zip(*(sequence[index:]
               for index in range(n)))
    )

compute_ngrams(['A','B','C','D'],2)

def flatten_corpus(corpus):
    return ' '.join([document.strip() 
                     for document in corpus])

def get_top_ngrams(corpus, ngram_val=1, limit=5):
    
    corpus = flatten_corpus(corpus)
    tokens = nltk.word_tokenize(corpus)

    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(), 
                              key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq) 
                     for text, freq in sorted_ngrams]

    return sorted_ngrams

get_top_ngrams(corpus=norm_alice, ngram_val=2,
               limit=10)

get_top_ngrams(corpus=norm_alice, ngram_val=3,
               limit=10)

### Collocations (Bigrams)

from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

finder = BigramCollocationFinder.from_documents([item.split() 
                                                for item 
                                                in norm_alice])
finder

Apply frequency filter:

finder.apply_freq_filter(5)

## Inherit BigramAssocMeasures
class AugmentedBigramAssocMeasures(BigramAssocMeasures):
    @classmethod
    def dp_fwd(self, *marginals):
        """Scores bigrams using phi-square, the square of the Pearson correlation
        coefficient.
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
        """Scores bigrams using phi-square, the square of the Pearson correlation
        coefficient.
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

Collocations based on raw frequencies:

bigram_measures = AugmentedBigramAssocMeasures()                                                
finder.nbest(bigram_measures.raw_freq, 10)

Collocations based on PMI:

finder.nbest(bigram_measures.pmi, 10)


bigrams_pmi=finder.score_ngrams(bigram_measures.pmi)
bigrams_pmi[:10]

bigrams_dpfwd=finder.score_ngrams(bigram_measures.dp_fwd)
bigrams_dpfwd[:10]

bigrams_dpbwd=finder.score_ngrams(bigram_measures.dp_bwd)
bigrams_dpbwd[:10]

### Collocations (N-grams)

from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures

finder = TrigramCollocationFinder.from_documents([item.split() 
                                                for item 
                                                in norm_alice])

Trigrams based on raw frequencies:


trigram_measures = TrigramAssocMeasures()                                                
finder.nbest(trigram_measures.raw_freq, 10)

Trigrams based on PMI:

finder.nbest(trigram_measures.pmi, 10)


## Weighted Tag-based Phrase Extraction

- Chunks

Load data.

data = open(DEMO_DATA_ROOT+'/elephants.txt', 'r+').readlines()
sentences = nltk.sent_tokenize(data[0])
len(sentences)

Normalize texts.

norm_sentences = tn.normalize_corpus(sentences, text_lower_case=False, 
                                     text_stemming=False, text_lemmatization=False, stopword_removal=False)
norm_sentences[:3]

Define chunk-based tokenizer.

import itertools
stopwords = nltk.corpus.stopwords.words('english')

def get_chunks(sentences, grammar = r'NP: {<DT>? <JJ>* <NN.*>+}', stopword_list=stopwords):
    
    all_chunks = []
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    
    for sentence in sentences:
        
        tagged_sents = [nltk.pos_tag(nltk.word_tokenize(sentence))]   
        
        chunks = [chunker.parse(tagged_sent) 
                      for tagged_sent in tagged_sents]
        
        wtc_sents = [nltk.chunk.tree2conlltags(chunk)
                         for chunk in chunks]    
        
        flattened_chunks = list(
                            itertools.chain.from_iterable(
                                wtc_sent for wtc_sent in wtc_sents)
                           )
        
        valid_chunks_tagged = [(status, [wtc for wtc in chunk]) 
                                   for status, chunk 
                                       in itertools.groupby(flattened_chunks, 
                                                lambda word_pos_chunk: word_pos_chunk[2] != 'O')]
        
        valid_chunks = [' '.join(word.lower() 
                                for word, tag, chunk in wtc_group 
                                    if word.lower() not in stopword_list) 
                                        for status, wtc_group in valid_chunks_tagged
                                            if status]
                                            
        all_chunks.append(valid_chunks)
    
    return all_chunks

Get chunks from texts.

chunks = get_chunks(norm_sentences)
chunks

Weight chunks.

from gensim import corpora, models

def get_tfidf_weighted_keyphrases(sentences, 
                                  grammar=r'NP: {<DT>? <JJ>* <NN.*>+}',
                                  top_n=10):
    
    valid_chunks = get_chunks(sentences, grammar=grammar)
                                     
    dictionary = corpora.Dictionary(valid_chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in valid_chunks]
    
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    
    weighted_phrases = {dictionary.get(idx): value 
                           for doc in corpus_tfidf 
                               for idx, value in doc}
                            
    weighted_phrases = sorted(weighted_phrases.items(), 
                              key=itemgetter(1), reverse=True)
    weighted_phrases = [(term, round(wt, 3)) for term, wt in weighted_phrases]
    
    return weighted_phrases[:top_n]

get_tfidf_weighted_keyphrases(sentences=norm_sentences, top_n=30)

from gensim.summarization import keywords

key_words = keywords(data[0], ratio=1.0, scores=True, lemmatize=True)
[(item, round(score, 3)) for item, score in key_words][:25]

## Topic Modeling



## Document Summarization

## References

- Text Analytics with Python (2nd Ed.) Chapter 6