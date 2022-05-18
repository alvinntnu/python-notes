#!/usr/bin/env python
# coding: utf-8

# # Word2Vec

# In[1]:


DEMO_DATA_ROOT = "../../../RepositoryData/data"


# ## Training 

# In[2]:


from gensim.models import word2vec


# In[3]:


get_ipython().run_cell_magic('time', '', '\n# sentences = word2vec.Text8Corpus(\'../data/text8\') \n# model = word2vec.Word2Vec(sentences, size=200, hs=1)\n\n# ## It takes a few minutes to train the model (about 7min on my Mac)\nmodel = word2vec.Word2Vec.load(DEMO_DATA_ROOT+"/text8_model") ## load the pretrained model\nprint(model)\n')


# In[4]:


# with open('../data/text8', 'r') as f:
#     for i in range(2):
#         l = f.readline()
#         print(l)

# f = open('../data/text8', 'r')
# l=f.readline()
# print(l)
# f.close()


# ## Functionality of Word2Vec

# In[5]:


model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)[0]


# In[6]:


model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])


# In[7]:


model.wv['computer']


# ## Save Model

# In[8]:


#model.save("../data/text8_model")


# In[9]:


## Load model
#model = word2vec.Word2Vec.load("../data/text8_model")


# In[10]:


## identify the most that is the most semantically distant from the others from a word list
model.wv.doesnt_match("breakfast cereal dinner lunch".split())


# In[11]:


model.wv.similarity('woman', 'man')


# In[12]:


model.wv.similarity('woman', 'cereal')


# In[13]:


model.wv.distance('man', 'woman')


# In[14]:


## Save model keyed vector
#word_vectors = model.wv
#del model


# ## Evaluating Word Vectors

# In[15]:


import os, gensim
module_path = gensim.__path__[0]
#print(module_path)
print(os.path.join(module_path, 'test/test_data','wordsim353.tsv'))
model.wv.evaluate_word_pairs(os.path.join(gensim.__path__[0], 'test/test_data','wordsim353.tsv'))


# In[16]:


model.wv.accuracy(os.path.join(module_path, 'test/test_data', 'questions-words.txt'))[1]


# ## Loading Pre-trained Model

# In[17]:


# from gensim.models import KeyedVectors
# load the google word2vec model
# filename = 'GoogleNews-vectors-negative300.bin'
# model = KeyedVectors.load_word2vec_format(filename, binary=True)


