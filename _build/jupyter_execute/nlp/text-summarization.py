#!/usr/bin/env python
# coding: utf-8

# # Text Summarization

# In[48]:


DEMO_DATA_ROOT = "../../../RepositoryData/data"


# - Efficient ways to summarize the semantics of massive collections of documents
# - Three general methods:
#     - Keyphrase extraction
#     - Topic modeling
#     - Document summarization

# ## Keyphrase Extraction

# - N-grams
# - Collocations

# In[1]:


from nltk.corpus import gutenberg
import text_normalizer as tn
import nltk
from operator import itemgetter


# Loading corpus, Alice in the Wonderland.

# In[2]:


# load corpus
alice = gutenberg.sents(fileids='carroll-alice.txt')
# concatenate each word token of a sentence
alice = [' '.join(ts) for ts in alice]
# normalize text
# `filter()` removes tokens that are False after normalization
norm_alice = list(filter(None, 
                         tn.normalize_corpus(alice, text_lemmatization=False))) 


# Compare raw texts vs, noramlized texts:

# In[3]:


print(alice[0], '\n', norm_alice[0])


# ### N-grams

# A function to create n-grams.

# In[4]:


def compute_ngrams(sequence, n):
    return list(
        zip(*(sequence[index:]
               for index in range(n)))
    )


# In[5]:


compute_ngrams(['A','B','C','D'],2)


# In[6]:


def flatten_corpus(corpus):
    return ' '.join([document.strip() 
                     for document in corpus])


# In[7]:


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


# In[8]:


get_top_ngrams(corpus=norm_alice, ngram_val=2,
               limit=10)


# In[9]:


get_top_ngrams(corpus=norm_alice, ngram_val=3,
               limit=10)


# ### Collocations (Bigrams)

# In[143]:


from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures

finder = BigramCollocationFinder.from_documents([item.split() 
                                                for item 
                                                in norm_alice])
finder


# Apply frequency filter:

# In[144]:


finder.apply_freq_filter(5)


# In[145]:


## Inherit BigramAssocMeasures
class AugmentedBigramAssocMeasures(BigramAssocMeasures):
    @classmethod
    def dp_fwd(cls, *marginals):
        """Scores bigrams using delta P forward, the normalized 
        conditional prob of w2 given w1: p(w2/w1)-p(w2/_w1)
        This may be shown with respect to a contingency table::

                w1    ~w1
             ------ ------
         w2 | n_ii | n_oi | = n_xi
             ------ ------
        ~w2 | n_io | n_oo |
             ------ ------
             = n_ix        TOTAL = n_xx
        """
        
        n_ii, n_oi, n_io, n_oo  = cls._contingency(*marginals)

        return (n_ii/(n_ii+n_io)) - (n_oi/(n_oi+n_oo))

    @classmethod
    def dp_bwd(cls, *marginals):
        """Scores bigrams using delta P forward, the normalized 
        conditional prob of w1 given w2: p(w1/w2)-p(w1/_w2)
        This may be shown with respect to a contingency table::

                w1    ~w1
             ------ ------
         w2 | n_ii | n_oi | = n_xi
             ------ ------
        ~w2 | n_io | n_oo |
             ------ ------
             = n_ix        TOTAL = n_xx
        """
        
        n_ii, n_oi, n_io, n_oo  = cls._contingency(*marginals)

        return (n_ii/(n_ii+n_oi)) - (n_io/(n_io+n_oo))


# Collocations based on raw frequencies:

# In[146]:


bigram_measures = AugmentedBigramAssocMeasures()                                                
finder.nbest(bigram_measures.raw_freq, 10)


# Collocations based on PMI:

# In[129]:


finder.nbest(bigram_measures.pmi, 10)


# In[141]:


bigrams_pmi=finder.score_ngrams(bigram_measures.pmi)
bigrams_pmi[:10]


# In[147]:


bigrams_dpfwd=finder.score_ngrams(bigram_measures.dp_fwd)
bigrams_dpfwd[:10]


# In[43]:


bigrams_dpbwd=finder.score_ngrams(bigram_measures.dp_bwd)
bigrams_dpbwd[:10]


# ### Collocations (N-grams)

# In[45]:


from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures

finder = TrigramCollocationFinder.from_documents([item.split() 
                                                for item 
                                                in norm_alice])


# Trigrams based on raw frequencies:

# In[46]:


trigram_measures = TrigramAssocMeasures()                                                
finder.nbest(trigram_measures.raw_freq, 10)


# Trigrams based on PMI:

# In[47]:


finder.nbest(trigram_measures.pmi, 10)


# ## Weighted Tag-based Phrase Extraction

# - Chunks

# Load data.

# In[50]:


data = open(DEMO_DATA_ROOT+'/elephants.txt', 'r+').readlines()
sentences = nltk.sent_tokenize(data[0])
len(sentences)


# Normalize texts.

# In[51]:


norm_sentences = tn.normalize_corpus(sentences, text_lower_case=False, 
                                     text_stemming=False, text_lemmatization=False, stopword_removal=False)
norm_sentences[:3]


# Define chunk-based tokenizer.

# In[52]:


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


# Get chunks from texts.

# In[53]:


chunks = get_chunks(norm_sentences)
chunks


# Weight chunks.

# In[54]:


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


# In[55]:


get_tfidf_weighted_keyphrases(sentences=norm_sentences, top_n=30)


# In[57]:


from gensim.summarization import keywords

key_words = keywords(data[0], ratio=1.0, scores=True, lemmatize=True)
[(item, round(score, 3)) for item, score in key_words][:25]


# ## Topic Modeling

# Download corpus and unzip the corpus files.

# In[59]:


#!wget https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz


# In[64]:


#!tar -xzf nips12raw_str602.tgz


# ### Load  Data

# In[65]:


import os
import numpy as np
import pandas as pd

DATA_PATH = 'nipstxt/'
print(os.listdir(DATA_PATH))
folders = ["nips{0:02}".format(i) for i in range(0,13)]
# Read all texts into a list.
papers = []
for folder in folders:
    file_names = os.listdir(DATA_PATH + folder)
    for file_name in file_names:
        with open(DATA_PATH + folder + '/' + file_name, encoding='utf-8', errors='ignore', mode='r+') as f:
            data = f.read()
        papers.append(data)
len(papers)


# In[66]:


print(papers[0][:100])


# ### Preprocess Data

# In[67]:


get_ipython().run_cell_magic('time', '', "import nltk\n\nstop_words = nltk.corpus.stopwords.words('english')\nwtk = nltk.tokenize.RegexpTokenizer(r'\\w+')\nwnl = nltk.stem.wordnet.WordNetLemmatizer()\n\ndef normalize_corpus(papers):\n    norm_papers = []\n    for paper in papers:\n        paper = paper.lower()\n        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]\n        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]\n        paper_tokens = [token for token in paper_tokens if len(token) > 1]\n        paper_tokens = [token for token in paper_tokens if token not in stop_words]\n        paper_tokens = list(filter(None, paper_tokens))\n        if paper_tokens:\n            norm_papers.append(paper_tokens)\n            \n    return norm_papers\n    \nnorm_papers = normalize_corpus(papers)\nprint(len(norm_papers))\n")


# ## Feature Enginerring

# In[68]:


import gensim

bigram = gensim.models.Phrases(norm_papers, min_count=20, threshold=20, delimiter=b'_') # higher threshold fewer phrases.
bigram_model = gensim.models.phrases.Phraser(bigram)

print(bigram_model[norm_papers[0]][:50])


# In[69]:


norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]

# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
print('Sample word to number mappings:', list(dictionary.items())[:15])
print('Total Vocabulary Size:', len(dictionary))


# In[70]:


# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)
print('Total Vocabulary Size:', len(dictionary))


# In[71]:


# Transforming corpus into bag of words vectors
bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
print(bow_corpus[1][:50])


# In[72]:


print([(dictionary[idx] , freq) for idx, freq in bow_corpus[1][:50]])


# In[73]:


print('Total number of papers:', len(bow_corpus))


# ### Topic Models with Latent Semantic Indexing (LSI)

# In[74]:


TOTAL_TOPICS = 10
lsi_bow = gensim.models.LsiModel(bow_corpus, id2word=dictionary, num_topics=TOTAL_TOPICS,
                                 onepass=True, chunksize=1740, power_iters=1000)


# In[75]:


for topic_id, topic in lsi_bow.print_topics(num_topics=10, num_words=20):
    print('Topic #'+str(topic_id+1)+':')
    print(topic)
    print()


# In[76]:


for n in range(TOTAL_TOPICS):
    print('Topic #'+str(n+1)+':')
    print('='*50)
    d1 = []
    d2 = []
    for term, wt in lsi_bow.show_topic(n, topn=20):
        if wt >= 0:
            d1.append((term, round(wt, 3)))
        else:
            d2.append((term, round(wt, 3)))

    print('Direction 1:', d1)
    print('-'*50)
    print('Direction 2:', d2)
    print('-'*50)
    print()


# In[77]:


term_topic = lsi_bow.projection.u
singular_values = lsi_bow.projection.s
topic_document = (gensim.matutils.corpus2dense(lsi_bow[bow_corpus], len(singular_values)).T / singular_values).T
term_topic.shape, singular_values.shape, topic_document.shape


# In[78]:


document_topics = pd.DataFrame(np.round(topic_document.T, 3), 
                               columns=['T'+str(i) for i in range(1, TOTAL_TOPICS+1)])
document_topics.head(15)


# In[79]:


document_numbers = [13, 250, 500]

for document_number in document_numbers:
    top_topics = list(document_topics.columns[np.argsort(-np.absolute(document_topics.iloc[document_number].values))[:3]])
    print('Document #'+str(document_number)+':')
    print('Dominant Topics (top 3):', top_topics)
    print('Paper Summary:')
    print(papers[document_number][:500])
    print()


# ### Topic Models with Latent Dirichlet Allocation (LDA)

# In[80]:


get_ipython().run_cell_magic('time', '', "lda_model = gensim.models.LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=1740, \n                                   alpha='auto', eta='auto', random_state=42,\n                                   iterations=500, num_topics=TOTAL_TOPICS, \n                                   passes=20, eval_every=None)\n")


# In[81]:


for topic_id, topic in lda_model.print_topics(num_topics=10, num_words=20):
    print('Topic #'+str(topic_id+1)+':')
    print(topic)
    print()


# In[82]:


topics_coherences = lda_model.top_topics(bow_corpus, topn=20)
avg_coherence_score = np.mean([item[1] for item in topics_coherences])
print('Avg. Coherence Score:', avg_coherence_score)


# In[83]:


topics_with_wts = [item[0] for item in topics_coherences]
print('LDA Topics with Weights')
print('='*50)
for idx, topic in enumerate(topics_with_wts):
    print('Topic #'+str(idx+1)+':')
    print([(term, round(wt, 3)) for wt, term in topic])
    print()


# In[84]:


print('LDA Topics without Weights')
print('='*50)
for idx, topic in enumerate(topics_with_wts):
    print('Topic #'+str(idx+1)+':')
    print([term for wt, term in topic])
    print()


# In[85]:


cv_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, 
                                                      texts=norm_corpus_bigrams,
                                                      dictionary=dictionary, 
                                                      coherence='c_v')
avg_coherence_cv = cv_coherence_model_lda.get_coherence()

umass_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, 
                                                         texts=norm_corpus_bigrams,
                                                         dictionary=dictionary, 
                                                         coherence='u_mass')
avg_coherence_umass = umass_coherence_model_lda.get_coherence()

perplexity = lda_model.log_perplexity(bow_corpus)

print('Avg. Coherence Score (Cv):', avg_coherence_cv)
print('Avg. Coherence Score (UMass):', avg_coherence_umass)
print('Model Perplexity:', perplexity)


# ### LDA with Mallet

# In[90]:


get_ipython().system('wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip')


# In[91]:


get_ipython().system('unzip -q mallet-2.0.8.zip')


# In[92]:


MALLET_PATH = 'mallet-2.0.8/bin/mallet'
lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path=MALLET_PATH, corpus=bow_corpus, 
                                              num_topics=TOTAL_TOPICS, id2word=dictionary,
                                              iterations=500, workers=16)


# In[93]:


topics = [[(term, round(wt, 3)) 
               for term, wt in lda_mallet.show_topic(n, topn=20)] 
                   for n in range(0, TOTAL_TOPICS)]

for idx, topic in enumerate(topics):
    print('Topic #'+str(idx+1)+':')
    print([term for term, wt in topic])
    print()


# In[94]:


cv_coherence_model_lda_mallet = gensim.models.CoherenceModel(model=lda_mallet, corpus=bow_corpus, 
                                                             texts=norm_corpus_bigrams,
                                                             dictionary=dictionary, 
                                                             coherence='c_v')
avg_coherence_cv = cv_coherence_model_lda_mallet.get_coherence()

umass_coherence_model_lda_mallet = gensim.models.CoherenceModel(model=lda_mallet, corpus=bow_corpus, 
                                                                texts=norm_corpus_bigrams,
                                                                dictionary=dictionary,  
                                                                coherence='u_mass')
avg_coherence_umass = umass_coherence_model_lda_mallet.get_coherence()

# from STDOUT: <500> LL/token: -8.53533
perplexity = -8.53533
print('Avg. Coherence Score (Cv):', avg_coherence_cv)
print('Avg. Coherence Score (UMass):', avg_coherence_umass)
print('Model Perplexity:', perplexity)


# ### LDA Tuning - Finading Optimal Number of Topics

# In[95]:


from tqdm import tqdm

def topic_model_coherence_generator(corpus, texts, dictionary, 
                                    start_topic_count=2, end_topic_count=10, step=1,
                                    cpus=1):
    
    models = []
    coherence_scores = []
    for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
        mallet_lda_model = gensim.models.wrappers.LdaMallet(mallet_path=MALLET_PATH, corpus=corpus,
                                                            num_topics=topic_nums, id2word=dictionary,
                                                            iterations=500, workers=cpus)
        cv_coherence_model_mallet_lda = gensim.models.CoherenceModel(model=mallet_lda_model, corpus=corpus, 
                                                                     texts=texts, dictionary=dictionary, 
                                                                     coherence='c_v')
        coherence_score = cv_coherence_model_mallet_lda.get_coherence()
        coherence_scores.append(coherence_score)
        models.append(mallet_lda_model)
    
    return models, coherence_scores


# In[96]:


lda_models, coherence_scores = topic_model_coherence_generator(corpus=bow_corpus, texts=norm_corpus_bigrams,
                                                               dictionary=dictionary, start_topic_count=2,
                                                               end_topic_count=30, step=1, cpus=16)


# In[97]:


coherence_df = pd.DataFrame({'Number of Topics': range(2, 31, 1),
                             'Coherence Score': np.round(coherence_scores, 4)})
coherence_df.sort_values(by=['Coherence Score'], ascending=False).head(10)


# In[98]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

x_ax = range(2, 31, 1)
y_ax = coherence_scores
plt.figure(figsize=(12, 6))
plt.plot(x_ax, y_ax, c='r')
plt.axhline(y=0.535, c='k', linestyle='--', linewidth=2)
plt.rcParams['figure.facecolor'] = 'white'
xl = plt.xlabel('Number of Topics')
yl = plt.ylabel('Coherence Score')


# In[99]:


best_model_idx = coherence_df[coherence_df['Number of Topics'] == 20].index[0]
best_lda_model = lda_models[best_model_idx]
best_lda_model.num_topics


# In[100]:


topics = [[(term, round(wt, 3)) 
               for term, wt in best_lda_model.show_topic(n, topn=20)] 
                   for n in range(0, best_lda_model.num_topics)]

for idx, topic in enumerate(topics):
    print('Topic #'+str(idx+1)+':')
    print([term for term, wt in topic])
    print()


# In[101]:


topics_df = pd.DataFrame([[term for term, wt in topic] 
                              for topic in topics], 
                         columns = ['Term'+str(i) for i in range(1, 21)],
                         index=['Topic '+str(t) for t in range(1, best_lda_model.num_topics+1)]).T
topics_df


# In[102]:


pd.set_option('display.max_colwidth', -1)
topics_df = pd.DataFrame([', '.join([term for term, wt in topic])  
                              for topic in topics],
                         columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, best_lda_model.num_topics+1)]
                         )
topics_df


# ### Interpreting Topic Model Results
# 

# In[103]:


tm_results = best_lda_model[bow_corpus]


# In[104]:


corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] 
                     for topics in tm_results]
corpus_topics[:5]


# In[105]:


corpus_topic_df = pd.DataFrame()
corpus_topic_df['Document'] = range(0, len(papers))
corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
corpus_topic_df['Topic Desc'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]
corpus_topic_df['Paper'] = papers


# Dominant topics distribution across corpus.

# In[112]:


# pd.set_option('display.max_colwidth', 200)
# topic_stats_df = corpus_topic_df.groupby('Dominant Topic').agg({
#                                                 'Dominant Topic': {
#                                                     'Doc Count': np.size,
#                                                     '% Total Docs': np.size }
#                                               })
# topic_stats_df = topic_stats_df['Dominant Topic'].reset_index()
# topic_stats_df['% Total Docs'] = topic_stats_df['% Total Docs'].apply(lambda row: round((row*100) / len(papers), 2))
# topic_stats_df['Topic Desc'] = [topics_df.iloc[t]['Terms per Topic'] for t in range(len(topic_stats_df))]
# topic_stats_df


# Dominant topics in specific papers

# In[107]:


pd.set_option('display.max_colwidth', 200)
(corpus_topic_df[corpus_topic_df['Document']
                 .isin([681, 9, 392, 1622, 17, 
                        906, 996, 503, 13, 733])])


# Relevant papers per topic based on dominance.

# In[114]:


corpus_topic_df.groupby('Dominant Topic').apply(lambda 
                                                topic_set: (topic_set.sort_values(by=['Contribution %'], 
                                                                                         ascending=False)))


# ### Predicting Topics for New Paper

# In[119]:


import glob
# papers manually downloaded from NIPS 16
# https://papers.nips.cc/book/advances-in-neural-information-processing-systems-29-2016

new_paper_files = glob.glob('test_data/nips16*.txt')
new_papers = []
for fn in new_paper_files:
    with open(fn, encoding='utf-8', errors='ignore', mode='r+') as f:
        data = f.read()
        new_papers.append(data)
              
print('Total New Papers:', len(new_papers))


# In[120]:


def text_preprocessing_pipeline(documents, normalizer_fn, bigram_model):
    norm_docs = normalizer_fn(documents)
    norm_docs_bigrams = bigram_model[norm_docs]
    return norm_docs_bigrams

def bow_features_pipeline(tokenized_docs, dictionary):
    paper_bow_features = [dictionary.doc2bow(text) 
                              for text in tokenized_docs]
    return paper_bow_features

norm_new_papers = text_preprocessing_pipeline(documents=new_papers, normalizer_fn=normalize_corpus, 
                                              bigram_model=bigram_model)
norm_bow_features = bow_features_pipeline(tokenized_docs=norm_new_papers, dictionary=dictionary)


# In[121]:


print(norm_new_papers[0][:30])


# In[122]:


print(norm_bow_features[0][:30])


# In[123]:


def get_topic_predictions(topic_model, corpus, topn=3):
    topic_predictions = topic_model[corpus]
    best_topics = [[(topic, round(wt, 3)) 
                        for topic, wt in sorted(topic_predictions[i], 
                                                key=lambda row: -row[1])[:topn]] 
                            for i in range(len(topic_predictions))]
    return best_topics


# In[124]:


topic_preds = get_topic_predictions(topic_model=best_lda_model, 
                                    corpus=norm_bow_features, topn=2)
topic_preds


# In[125]:


results_df = pd.DataFrame()
results_df['Papers'] = range(1, len(new_papers)+1)
results_df['Dominant Topics'] = [[topic_num+1 for topic_num, wt in item] for item in topic_preds]
res = results_df.set_index(['Papers'])['Dominant Topics'].apply(pd.Series).stack().reset_index(level=1, drop=True)
results_df = pd.DataFrame({'Dominant Topics': res.values}, index=res.index)
results_df['Contribution %'] = [topic_wt for topic_list in 
                                        [[round(wt*100, 2) 
                                              for topic_num, wt in item] 
                                                 for item in topic_preds] 
                                    for topic_wt in topic_list]

results_df['Topic Desc'] = [topics_df.iloc[t-1]['Terms per Topic'] for t in results_df['Dominant Topics'].values]
results_df['Paper Desc'] = [new_papers[i-1][:200] for i in results_df.index.values]


# In[126]:


pd.set_option('display.max_colwidth', 300)
results_df


# ## Document Summarization

# ## References
# 
# - Text Analytics with Python (2nd Ed.) Chapter 6
