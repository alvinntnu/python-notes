# Chinese Word Segmentation (jieba)


## Important Steps

- Install `jieba` module

```
!pip install jieba
```

- import module

```
import jieba
import jieba.analyse
```

- initialize traditional Chinese dictionary
    - Download the traditional chinese dictionary from [`jieba-tw`](https://raw.githubusercontent.com/ldkrsi/jieba-zh_TW/master/jieba/dict.txt)
    
```
jieba.set_dictionary(file_path)
```

- Add own project-specific dictionary

```
jieba.load_userdict(file_path)
```

- Add add-hoc words to dictionary

```
jieba.add_word(word, freq=None, tag=None)
```

- Remove words

```
jieba.del_word(word)
```

- Chinese stopwords (See [林宏任老師 GitHub](https://github.com/tomlinNTUB/Python/tree/master/%E4%B8%AD%E6%96%87%E5%88%86%E8%A9%9E)

    - `jieba.cut()` does not interact with stopword list
    - `jieba.analyse.set_stop_words(file_apth)`

- Word segmentation
    - `jieba.cut()` returns a `generator` object
    - `jieba.lcut()` resuts a `List` object
    
```
# full

jieba.cut(TEXT, cut_all=True)
jieba.lcut(TEXT, cut_all=True

# default
jieba.cut(TEXT, cut_all=False)
jieba.lcut(TEXT, cut_all=False)
```

- Keyword Extraction
    - The module uses the TF-IDF score to extract keywords
    - But how documents are defined in Jieba? Eahc list element in the input is a doc?

```
jieba.analyse.extract_tags(TEXT, topK=20, withWeight=False, allowPOS=())
```


    

## Demonstration

import jieba
from jieba import posseg

# set dictionary

jieba.set_dictionary('../../../Corpus/jiaba/dict.txt.jiebatw.txt/')
#jieba.load_userdict()

text = '據《日經亞洲評論》網站報導，儘管美國總統川普發起了讓美國製造業回歸的貿易戰，但包括電動汽車製造商特斯拉在內的一些公司反而加大馬力在大陸進行生產。另據高盛近日發布的一份報告指出，半導體設備和材料以及醫療保健領域的大多數公司實際上正擴大在大陸的生產，許多美國製造業拒絕「退出中國」。'

print(' '.join(jieba.cut(text, cut_all=False, HMM=True))+'\n')
print(' '.join(jieba.cut(text, cut_all=False, HMM=False))+'\n')
print(' '.join(jieba.cut(text, cut_all=True, HMM=True))+'\n')

text_pos = posseg.cut(text)
#print(type(text_pos))
for word, tag in text_pos:
    print(word+'/'+tag)

# load stopwords
with open('../../../Corpus/stopwords/tomlinNTUB-chinese-stopwords.txt', 'r') as f:
    stopwords = [w.strip() for w in f.readlines()]

words1 = jieba.lcut(text, cut_all=True)
words2 = [w for w in words1 if w not in stopwords]

print(len(words1))
print(len(words2))
print(words2)

## Word Cloud

from collections import Counter

wf = dict(sorted(Counter(words2).items(), key=lambda x:x[1], reverse=True))


import matplotlib.pyplot as plt
from wordcloud import WordCloud
wc = WordCloud(background_color='white',
               font_path='/System/Library/Fonts/STHeiti Medium.ttc',
               random_state=10,
               max_font_size=None,
               stopwords=['包括']) ## stopwords not work when wc.genreate_from_frequencies
wc.generate_from_frequencies(frequencies=wf)

plt.figure(figsize=(15,15))
plt.imshow(wc)
plt.axis("off")
plt.show()

#wc.to_file(FILE_PATH)