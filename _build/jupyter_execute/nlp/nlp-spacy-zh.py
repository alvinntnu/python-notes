#!/usr/bin/env python
# coding: utf-8

# # Chinese Natural Language Processing (spaCy)

# ## Installation
# 
# ```
# # Install package
# ## In terminal:
# !pip install spacy
# 
# ## Download language model for Chinese and English
# !spacy download en
# !python -m spacy download zh_core_web_lg
# ```
# 
# - Documentation: [spaCy Chinese model](https://github.com/howl-anderson/Chinese_models_for_SpaCy)

# In[1]:


import spacy
from spacy import displacy
# load language model
nlp_zh = spacy.load('zh_core_web_sm')## disable=["parser"]
# parse text 
doc = nlp_zh('這是一個中文的句子')


# ## Linguistic Features
# 
# - After we parse and tag a given text, we can extract token-level information:
#     - Text: the original word text
#     - Lemma: the base form of the word
#     - POS: the simple universal POS tag
#     - Tag: the detailed POS tag
#     - Dep: Syntactic dependency
#     - Shape: Word shape (capitalization, punc, digits)
#     - is alpha
#     - is stop
#     
# :::{note}
# :class: dropdown
# For more information on POS tags, see spaCy (POS tag scheme documentation)[https://spacy.io/api/annotation#pos-tagging].
# :::

# In[2]:


# parts of speech tagging
for token in doc:
    print(((token.text, 
            token.lemma_, 
            token.pos_, 
            token.tag_,
            token.dep_,
            token.shape_,
            token.is_alpha,
            token.is_stop,
            )))


# In[ ]:


## Output in different ways
for token in doc:
    print('%s_%s' % (token.text, token.tag_))
    
out = ''
for token in doc:
    out = out + ' '+ '/'.join((token.text, token.tag_))
print(out)


# In[48]:


# Noun chunking not working??
for n in doc.noun_chunks:
    print(n.text)


# In[5]:


## Check meaning of a POS tag (Not working??)
spacy.explain('VC')


# ## Visualization Linguistic Features

# In[6]:


# Visualize
displacy.render(doc, style="dep")


# In[9]:


options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro",
          "distance": 120}
displacy.render(doc, style="dep", options=options)


# In[22]:


## longer paragraphs
text_long = '''武漢肺炎全球肆虐，至今已有2906萬人確診、92萬染疫身亡，而流亡美國的中國大陸病毒學家閻麗夢，14日時開通了推特帳號，並公布一份長達26頁的科學論文，研究直指武肺病毒與自然人畜共通傳染病的病毒不同，並呼籲追查武漢P4實驗室及美國衛生研究院（NIH）之間的金流，引發討論。'''
text_long_list = text_long.split(sep="，")
len(text_long_list)

for c in text_long_list:
    print(c)


# In[23]:


## parse the texts
doc2 = list(nlp_zh.pipe(text_long_list))
len(doc2)


# In[27]:





# In[28]:


# Visual dependency for each sentence-like chunk
sentence_spans = list(doc2)
options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro",
          "distance": 120}
displacy.render(sentence_spans, style="dep", options=options)


# In[29]:


colors = {"CARDINAL": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
options = {"ents": ["CARDINAL"], "colors": colors}


displacy.render(sentence_spans[1], style="ent")


# ## NP Chunking

# In[49]:


## Print noun phrase for each doc
for d in doc2:
    for np in d.noun_chunks:
        print(np.text, 
              np.root.text,
              np.root.dep_,
              np.root.head.text)
    print('---')


# ## Named Entity Recognition
# 
# - Text: original entity text
# - Start: index of start of entity in the Doc
# - End: index of end of entity in the Doc
# - Label: Entity label, type
# 

# In[36]:


## Print ents for each doc
for d in doc2:
    for e in d.ents:
        print(e.text, e.label_)
    print('---')

