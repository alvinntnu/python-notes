#!/usr/bin/env python
# coding: utf-8

# # Language Model

# - Create the traditinal ngram-based language model
# - Codes from [A comprehensive guide to build your own language model in python](https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d)

# ## Training a Trigram Language Model using Reuters

# In[15]:


get_ipython().run_cell_magic('time', '', "\n# code courtesy of https://nlpforhackers.io/language-models/\n\nfrom nltk.corpus import reuters\nfrom nltk import bigrams, trigrams\nfrom collections import Counter, defaultdict\n\n# Create a placeholder for model\nmodel = defaultdict(lambda: defaultdict(lambda: 0))\n\n# Count frequency of co-occurance  \nfor sentence in reuters.sents():\n    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):\n        model[(w1, w2)][w3] += 1\n \n# Let's transform the counts to probabilities\nfor w1_w2 in model:\n    total_count = float(sum(model[w1_w2].values()))\n    for w3 in model[w1_w2]:\n        model[w1_w2][w3] /= total_count\n")


# ## Check Language Model

# In[12]:


sorted(dict(model["the","news"]).items(), key=lambda x:-1*x[1])


# ## Text Generation Using the Trigram Model

# - Using the trigram model to predict the next word.
# - The prediction is based on the predicted probability distribution of the next words: words above a predefined cut-off are randomly selected.
# - The text generator ends when two consecutuve None's are predicted (signaling the end of the sentence).

# In[16]:


# code courtesy of https://nlpforhackers.io/language-models/
import random

# starting words
text = ["the", "news"]
sentence_finished = False
 
while not sentence_finished:
  # select a random probability threshold  
  r = random.random()
  accumulator = .0

  for word in model[tuple(text[-2:])].keys():
      accumulator += model[tuple(text[-2:])][word]
      # select words that are above the probability threshold
      if accumulator >= r:
          text.append(word)
          break

  if text[-2:] == [None, None]:
      sentence_finished = True
 
print (' '.join([t for t in text if t]))


# ## Issues of Ngram Language Model

# - The ngram size is of key importance. The higher the order of the ngram, the better the prediction. But it comes with the issues of computation overload and data sparceness.
# - Unseen ngrams are always a concern.
# - Probability smoothing issues.
# 

# ## Neural Language Model

# - Neural language model based on deep learning may provide a better alternative to model the probabilistic relationships of linguistic units.
