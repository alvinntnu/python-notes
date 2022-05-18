#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing (ckipnlp)
# 
# - Chinese NLP toolkit developed by Academia Sinica
# - The CPU version works pretty slowly
# - The documentation of `ckipnlp` is limited. Need more time to figure out what is what and how to do what :)
# 
# - Documentation:
#     - [ckipnlp](https://ckipnlp.readthedocs.io)
# 

# In[3]:


from ckipnlp.pipeline import CkipPipeline, CkipDocument

pipeline = CkipPipeline()
doc = CkipDocument(raw='中研院的開發系統，來測試看看，挺酷的！')


# In[4]:


# Word Segmentation
pipeline.get_ws(doc)
print(doc.ws)
for line in doc.ws:
    print(line.to_text())

# Part-of-Speech Tagging
pipeline.get_pos(doc)
print(doc.pos)
for line in doc.pos:
    print(line.to_text())

# Named-Entity Recognition
pipeline.get_ner(doc)
print(doc.ner)
# Constituency Parsing
#pipeline.get_conparse(doc)
#print(doc.conparse)


# In[5]:


from ckipnlp.container.util.wspos import WsPosParagraph

# Word Segmentation & Part-of-Speech Tagging
for line in WsPosParagraph.to_text(doc.ws, doc.pos):
    print(line)


# In[6]:


from ckipnlp.container.util.wspos import WsPosSentence
for line in WsPosParagraph.to_text(doc.ws, doc.pos):
    print(line)


# In[7]:


doc2 = CkipDocument(raw='武漢肺炎全球肆虐，至今已有2906萬人確診、92萬染疫身亡，而流亡美國的中國大陸病毒學家閻麗夢，14日時開通了推特帳號，並公布一份長達26頁的科學論文，研究直指武肺病毒與自然人畜共通傳染病的病毒不同，並呼籲追查武漢P4實驗室及美國衛生研究院（NIH）之間的金流，引發討論。')
# Word Segmentation & Part-of-Speech Tagging

# Word Segmentation
pipeline.get_ws(doc2)
print(doc2.ws)
for line in doc2.ws:
    print(line.to_text())


# In[8]:


# Part-of-Speech Tagging
pipeline.get_pos(doc2)
print(doc2.pos)
for line in doc2.pos:
    print(line.to_text())


# In[9]:


for line in WsPosParagraph.to_text(doc2.ws, doc2.pos):
    print(line)


# In[10]:


pipeline.get_ner(doc2)
print(doc2.ner)

#WsPosSentence.to_text(doc2.ws, doc2.pos)

