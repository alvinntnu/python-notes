# Word Embeddings

- A simple walk-through of word2vec on a Chinese data of presidential inaugaural addresses


## Loading Corpus Raw Texts

import nltk
from nltk.corpus.reader import PlaintextCorpusReader
import numpy as np
import jieba, re

jieba.set_dictionary("../../../Corpus/jiaba/dict.txt.big.txt")

corpus_dir = "../../../Corpus/TaiwanPresidentialInaugarationSpeech_en/"

twp = PlaintextCorpusReader(corpus_dir, ".*\.txt")

len(twp.raw())


## Word Segmentation (ckiptagger)

from ckiptagger import WS

ws = WS("../../../Corpus/CKIP_WordSeg/data/")

## Print first 200 chars of file 13
print(twp.raw(fileids=twp.fileids()[13])[:200])

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

The following experiments try to see whether a few parameters may impact the performance of Chinese tokenization:

1. Segmenter: `ckiptagger` vs. `jibea`
2. Data Structure: `List` vs. `numpy.array`

It seems that `jieba` with `List` structure is the fastest?

twp_corpus = np.array([twp.raw(fileids=fid) for fid in twp.fileids()])
twp_corpus_list = [twp.raw(fileids=fid) for fid in twp.fileids()]

%%time
twp_corpus_seg1a = tokenize_corpus1(twp_corpus)

%%time
twp_corpus_seg1b = tokenize_corpus1(twp_corpus_list)

%%time
twp_corpus_seg3a = tokenize_corpus3(twp_corpus)

%%time
twp_corpus_seg3b = tokenize_corpus3(twp_corpus_list)

twp_corpus[13,][:200]

twp_corpus_seg1a[13][:200]

twp_corpus_seg3a[13][:200]

## Data Frame Representation

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

## choose one version of segmented texts
twp_corpus_seg = twp_corpus_seg3a

wst =nltk.WhitespaceTokenizer()
tokenized_corpus = [wst.tokenize(text) for text in twp_corpus_seg]

## Concordance

twp_text = nltk.text.Text(sum(tokenized_corpus,[]))
twp_text.concordance('台灣')

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
with open('../../../Corpus/stopwords/tomlinNTUB-chinese-stopwords.txt') as f:
    stopwords = [w.strip() for w in f.readlines()]

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




back_color = imageio.imread('../image/tw-char.jpg')
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

fig_path = '../data/twp-wordcloud.png'
wordcloud.to_file(fig_path)

## Set features for parameters
embedding_size = 100
context_size = 20
min_word_count = 1
sample = 1e-3

%%time
from gensim.models import word2vec

w2v_model = word2vec.Word2Vec(tokenized_corpus, 
                              size=embedding_size,
                              window=context_size,
                              min_count=min_word_count,
                              sample=sample,
                              iter=50)

## View Similar Words
w2v_model.wv.most_similar('人民', topn=5)
w2v_model.wv.most_similar('台灣', topn=5)

similar_words = {key_word:[similar_word[0] for similar_word in w2v_model.wv.most_similar(key_word, topn=6)]
                          for key_word in ['台灣','人民','國家','民主','中共','大陸','共匪','自由']}
similar_words

## Visualization

from sklearn.manifold import TSNE
all_words = sum([[key_word]+similar_words for key_word, similar_words in similar_words.items()], [])
all_words_vec = w2v_model.wv[all_words]

tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(all_words_vec)
labels=all_words

## Chinese Font Issues in Plotting

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
# 解决负号'-'显示为方块的问题
# rcParams['axes.unicode_minus']=False
myfont = FontProperties(fname='/System/Library/Fonts/PingFang.ttc',
 size=12)
# plt.title('乘客等级分布', fontproperties=myfont)
# plt.ylabel('人数', fontproperties=myfont)
# plt.legend(('头等舱', '二等舱', '三等舱'), loc='best', prop=myfont)

import matplotlib.pyplot as plt
plt.figure(figsize=(18,10))
plt.scatter(T[:,0],T[:,1], c="orange", edgecolors='r')
for label,x,y in zip(labels, T[:,0],T[:,1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0,0), textcoords='offset points',fontproperties=myfont)