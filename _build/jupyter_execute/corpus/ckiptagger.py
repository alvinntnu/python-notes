#!/usr/bin/env python
# coding: utf-8

# # Chinese Word Segmentation (ckiptagger)
# 
# 

# In[3]:


DEMO_DATA_ROOT="../../../RepositoryData/data"


# The current state-of-art Chinese segmenter for Taiwan Mandarin available is probably the [CKIP tagger](https://github.com/ckiplab/ckiptagger), created by the [Chinese Knowledge and Information Processing (CKIP)](https://ckip.iis.sinica.edu.tw/) group at the Academia Sinica.
# 
# The `ckiptagger` is released as a python module.
# 
# The normal CPU version is very slow. Not sure if it is the case for GPU version.

# ## Installation
# 
# ```
# !pip install ckiptagger
# ```
# 
# ```{note}
# Remember to download the model files. Very big.
# ```

# ## Download the Model Files
# 
# All NLP applications have their models behind their fancy performances. To use the tagger provided in `ckiptagger`, we need to download their pre-trained model files. 
# 
# Please go to the [github of CKIP tagger](https://github.com/ckiplab/ckiptagger) to download the model files, which is provided as a zipped file. (The file is very big. It takes a while.)
# 
# After you download the zipped file, unzip it under your working directory to the `data/` directory.

# ## Segmenting Texts
# 
# The initialized word segmenter object, `ws()`, can tokenize any input **character vectors** into a list of **word vectors** of the same size.

# In[4]:


from ckiptagger import data_utils, construct_dictionary, WS, POS, NER


# ```{note}
# Please use own model path. The model files are very big. They are probably not on the Drive.
# ```

# In[6]:


# Set Parameter Path
MODEL_PATH = '../../../../../Dropbox/Corpus/CKIP_WordSeg/data/'
ws = WS(MODEL_PATH)
pos = POS(MODEL_PATH)
ner = NER(MODEL_PATH)


# In[7]:


## Raw text corpus 
sentence_list = ['傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。',
              '美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。',
              '土地公有政策?？還是土地婆有政策。',
              '… 你確定嗎… 不要再騙了……他來亂的啦',
              '最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.',
              '科長說:1,坪數對人數為1:3。2,可以再增加。']
    ## other parameters
    # sentence_segmentation = True, # To consider delimiters
    # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
    # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
    # coerce_dictionary = dictionary2, # words in this dictionary are forced

word_list = ws(sentence_list)
pos_list = pos(word_list)
entity_list = ner(word_list, pos_list)
    


# In[8]:


def print_word_pos_sentence(word_sentence, pos_sentence):
    assert len(word_sentence) == len(pos_sentence)
    for word, pos in zip(word_sentence, pos_sentence):
        print(f"{word}({pos})", end="\u3000")
    print()
    return
    
for i, sentence in enumerate(sentence_list):
    print()
    print(f"'{sentence}'")
    print_word_pos_sentence(word_list[i],  pos_list[i])
    for entity in sorted(entity_list[i]):
        print(entity)


# ## Define Own Dictionary
# 
# The performance of Chinese word segmenter depends highly on the dictionary. Texts in different disciplines may have very domain-specific vocabulary. To prioritize a set of words in a dictionary, we can further ensure the accuracy of the word segmentation.
# 
# To create a dictionary for `ckiptagger`:

# In[9]:


word_to_weight = {
    "土地公": 1,
    "土地婆": 1,
    "公有": 2,
    "": 1,
    "來亂的": "啦",
    "緯來體育台": 1,
}
dictionary = construct_dictionary(word_to_weight)
print(dictionary)

word_list_2 = ws(sentence_list,
                recommend_dictionary=dictionary)
print(word_list)
print(word_list_2)


# ## Convert ckiptagger output into a Data Frame?
