# Word Cloud

## Important Steps

- Prepare word frequency data
    - `List`: Word list
    - `dictionary`: Word frequency dictionary

- Determine filtering criteria
    - remove stopwords?
    - remove low-frequency words?

- Chinese font issues
    - Check system-supported Chinese fonts
    
    ```
    !fc-list :lang=zh  
    ```
    
    - Specify the font_path when initialize the `WordCloud`
    

- `WordCloud` Parameters (selective)

```
wc = WordCloud(
    font_path=..., # chinese font path
    width=...,
    height=...,
    margin=...,
    mask=...,
    max_words=...,
    min_font_size=4,
    max_font_size=None,
    stopwords=None, # a set with stopwords
    random_state=None,
    min_word_length=0
```
- Creating the word cloud from data
    - `WordCloud.generate()` expects a text (non-word-segmented long string of texts)
    - `WordCloud.generate_from_frequencies()` expects a dictionary of {word:freq}

- Increase Wordcloud resolution:
    - When intializing the WordCloud, specify the `width` and `heigth`
    - When plotting, specify the figure size of plt:
    ```
    plt.figure(figsize=(20,10), facecolor='k')
    ```
- Create a image-masked word cloud
    - Prepare a jpg of the mask (the white background will mask the words)

```
import imageio
from wordcloud import WordCloud, ImageColorGenerator

back_color = imageio.imread(IMAGE_PATH)

wordcloud = WordCloud(mask=back_color)

plt.figure(figsize=(20,10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

wordcloud.to_file(PATH)


# Use image color for words
image_colors = ImageColorGenerator(back_color)
plt.imshow(wordcloud.recolor(color_func=image_colors))
```

## Demonstration

- Extract the first article from Google News
- Tokenize the news
- Create the word cloud

## Prepare Text Data

import requests 
from bs4 import BeautifulSoup
import pandas as pd
 
 
url = 'https://news.google.com/topics/CAAqJQgKIh9DQkFTRVFvSUwyMHZNRFptTXpJU0JYcG9MVlJYS0FBUAE?hl=zh-TW&gl=TW&ceid=TW%3Azh-Hant'
r = requests.get(url)
web_content = r.text
soup = BeautifulSoup(web_content,'lxml')
title = soup.find_all('a', class_='DY5T1d')
first_art_link = title[0]['href'].replace('.','https://news.google.com',1)

#print(first_art_link)
art_request = requests.get(first_art_link)
art_request.encoding='utf8'
soup_art = BeautifulSoup(art_request.text,'lxml')

art_content = soup_art.find_all('p')
art_texts = [p.text for p in art_content]
print(art_texts)
## Create Word Cloud

import jieba

jieba.set_dictionary('../../../Corpus/jiaba/dict.txt.big.txt')

art_words = [w for w in jieba.cut(' '.join(art_texts))]
## Fine-tune Word Cloud

from collections import Counter
import imageio
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from matplotlib import pyplot as plt


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
wordcloud.generate(' '.join([w for w in art_words if w not in stopwords]))
plt.figure(figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
#plt.savefig('../data/twp-wordcloud2.png', facecolor='k', bbox_inches='tight')



## References

- [筆記 for Python (Jieba + Wordcloud)](https://medium.com/@fsflyingsoar/%E7%AD%86%E8%A8%98-for-python-jieba-wordcloud-b814f5e04e01)

- [以 jieba 與 gensim 探索文本主題：五月天人生無限公司歌詞分析 ( I )](https://medium.com/pyladies-taiwan/%E4%BB%A5-jieba-%E8%88%87-gensim-%E6%8E%A2%E7%B4%A2%E6%96%87%E6%9C%AC%E4%B8%BB%E9%A1%8C-%E4%BA%94%E6%9C%88%E5%A4%A9%E4%BA%BA%E7%94%9F%E7%84%A1%E9%99%90%E5%85%AC%E5%8F%B8%E6%AD%8C%E8%A9%9E%E5%88%86%E6%9E%90-i-cd2147b89083)