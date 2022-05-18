#!/usr/bin/env python
# coding: utf-8

# # Word Embeddings with Chinese Texts
# 
# - A simple walk-through of word2vec on a Chinese data of presidential inaugaural addresses
# 

# In[1]:


DEMO_DATA_ROOT = "../../../RepositoryData/data"


# ## Loading Corpus Raw Texts

# In[2]:


import nltk
from nltk.corpus.reader import PlaintextCorpusReader
import numpy as np
import jieba, re

jieba.set_dictionary(DEMO_DATA_ROOT + "/jiaba/dict.txt.big.txt")


# In[3]:


corpus_dir = DEMO_DATA_ROOT+"/TaiwanPresidentialInaugarationSpeech_en"

twp = PlaintextCorpusReader(corpus_dir, ".*\.txt")


# In[4]:


len(twp.raw())


# ## Word Segmentation
# 
# - Try two methods: `ckiptagger` vs. `jieba`

# In[5]:


from ckiptagger import WS


# ```{margin}
# ```{note}
# Please remember to download the CKIP model files and change the path accordingly.
# ```
# ```

# In[6]:


ws = WS("/Users/Alvin/Dropbox/Corpus/CKIP_WordSeg/data")


# In[7]:


## Print first 200 chars of file 13
print(twp.raw(fileids=twp.fileids()[13])[:200])


# In[8]:


# word-seg the raw text and return a long string
def tokenize_raw1(raw):
    word_tok = [' '.join(para) for para in ws(nltk.regexp_tokenize(raw, r'[^\s]+'))] # para-like units
    raw_tok  = ' '.join(word_tok)
    return raw_tok

# word-seg the raw text and return list of words
def tokenize_raw2(raw):
    para_list = nltk.regexp_tokenize(raw, r'[^\s]+') # para-like units
    word_list = sum(ws(para_list),[]) 
    return word_list


def tokenize_raw3(raw):
    raw = re.sub(r'[\n\s\r]+', '', raw)
    return ' '.join([w for w in jieba.cut(raw)])

tokenize_corpus1 = np.vectorize(tokenize_raw1)
tokenize_corpus2 = np.vectorize(tokenize_raw2)
tokenize_corpus3 = np.vectorize(tokenize_raw3)


# The following experiments try to see whether a few parameters may impact the performance of Chinese tokenization:
# 
# 1. Segmenter: `ckiptagger` vs. `jibea`
# 2. Data Structure: `List` vs. `numpy.array`
# 
# It seems that `jieba` with `List` structure is the fastest?

# In[9]:


twp_corpus = np.array([twp.raw(fileids=fid) for fid in twp.fileids()])
twp_corpus_list = [twp.raw(fileids=fid) for fid in twp.fileids()]


# In[10]:


get_ipython().run_cell_magic('time', '', 'twp_corpus_seg1a = tokenize_corpus1(twp_corpus)\n')


# In[11]:


get_ipython().run_cell_magic('time', '', 'twp_corpus_seg1b = tokenize_corpus1(twp_corpus_list)\n')


# In[12]:


get_ipython().run_cell_magic('time', '', 'twp_corpus_seg3a = tokenize_corpus3(twp_corpus)\n')


# In[13]:


get_ipython().run_cell_magic('time', '', 'twp_corpus_seg3b = tokenize_corpus3(twp_corpus_list)\n')


# In[14]:


twp_corpus[13,][:200]


# In[15]:


twp_corpus_seg1a[13][:200]


# In[16]:


twp_corpus_seg3a[13][:200]


# ## Data Frame Representation

# In[17]:


## data frame representation
import pandas as pd
import re


twp_df = pd.DataFrame({
    "fileid": twp.fileids(),
    "corpus_raw": twp_corpus,
    "corpus_seg_ckip": twp_corpus_seg1a,
    "corpus_seg_jb": twp_corpus_seg3a
})
twp_df[['year','id','president']] = twp_df['fileid'].str.split('_', expand=True)
twp_df['president']=twp_df['president'].str.replace('.txt','')
twp_df


# ## Word Cloud

# In[18]:


## choose one version of segmented texts
twp_corpus_seg = twp_corpus_seg3a


# In[19]:


wst =nltk.WhitespaceTokenizer()
tokenized_corpus = [wst.tokenize(text) for text in twp_corpus_seg]


# In[20]:


## Concordance

twp_text = nltk.text.Text(sum(tokenized_corpus,[]))
twp_text.concordance('台灣')


# In[23]:


from collections import Counter
import imageio
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from matplotlib import pyplot as plt

words = sum(tokenized_corpus,[])
words[:5]
sorted(Counter(words).items(), key=lambda x:x[1], reverse=True)  



## Check font paths
## !fc-list :lang=zh


## Load stopwords
with open(DEMO_DATA_ROOT+'/stopwords/tomlinNTUB-chinese-stopwords.txt') as f:
    stopwords = [w.strip() for w in f.readlines()]


# In[24]:


# Create stopwords ad hoc
stopwords = set(['一個'])
wordcloud = WordCloud(font_path="/System/Library/Fonts/PingFang.ttc",
                      background_color='black',
                      #stopwords=stopwords,
                      width=1600, height=800 
                     )  ##add  system chinese font path
wordcloud.generate(' '.join([w for w in words if w not in stopwords]))
plt.figure(figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
#plt.savefig('../data/twp-wordcloud2.png', facecolor='k', bbox_inches='tight')



# In[28]:


back_color = imageio.imread(DEMO_DATA_ROOT+ '/image/tw-char.jpg')
wordcloud = WordCloud(font_path="/System/Library/Fonts/PingFang.ttc",
                      background_color='black',
                      mask=back_color,
                      random_state=10,
                      #max_font_size=20,
                      #min_font_size=20,
                      width=1200,
                      height=800)
wordcloud.generate(' '.join([w for w in words if w not in stopwords]))
image_colors = ImageColorGenerator(back_color)
plt.figure(figsize=(20,10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

fig_path = 'twp-wordcloud.png'
wordcloud.to_file(fig_path)


# ## Creating Word Embeddings

# In[29]:


## Set features for parameters
embedding_size = 100
context_size = 20
min_word_count = 1
sample = 1e-3


# In[30]:


get_ipython().run_cell_magic('time', '', 'from gensim.models import word2vec\n\nw2v_model = word2vec.Word2Vec(tokenized_corpus, \n                              size=embedding_size,\n                              window=context_size,\n                              min_count=min_word_count,\n                              sample=sample,\n                              iter=50)\n')


# ## Exploring Semantic Similarities

# In[31]:


## View Similar Words
w2v_model.wv.most_similar('人民', topn=5)
w2v_model.wv.most_similar('台灣', topn=5)


# In[32]:


similar_words = {key_word:[similar_word[0] for similar_word in w2v_model.wv.most_similar(key_word, topn=6)]
                          for key_word in ['台灣','人民','國家','民主','中共','大陸','共匪','自由']}
similar_words


# ## Visualization

# In[33]:


## Visualization

from sklearn.manifold import TSNE
all_words = sum([[key_word]+similar_words for key_word, similar_words in similar_words.items()], [])
all_words_vec = w2v_model.wv[all_words]

tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(all_words_vec)
labels=all_words


# In[63]:


## Chinese Font Issues in Plotting

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# rcParams['axes.unicode_minus']=False
myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')


# In[73]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6), dpi=300)
plt.scatter(T[:,0],T[:,1], c="orange", edgecolors='r', alpha=0.7, s=10)
for label,x,y in zip(labels, T[:,0],T[:,1]):
    plt.annotate(label, xy=(x, y), xytext=(-20,0), size=8, textcoords='offset points',fontproperties=myfont)

