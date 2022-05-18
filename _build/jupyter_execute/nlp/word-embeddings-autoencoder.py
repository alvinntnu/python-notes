#!/usr/bin/env python
# coding: utf-8

# # Generate Text Embeddings Using AutoEncoder

# ## Preparing the Input

# In[2]:


import nltk
from nltk.corpus import brown
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Input, Model, optimizers
from keras.layers import Bidirectional, LSTM, Embedding, RepeatVector, Dense
import numpy as np


# In[3]:


nltk.download('brown')


# In[4]:


sents = brown.sents()


# In[5]:


len(sents)


# In[6]:


maxlen = max([len(s) for s in sents])


# In[7]:


print(maxlen)


# In[8]:


vocab = set(brown.words())


# In[9]:


num_words = len(vocab)
print(num_words)
print(len(sents))


# In[10]:


num_words = 10000
embed_dim = 128
batch_size = 512
maxlen = 60


# ## Tokenizing and Padding

# In[11]:


tokenizer = Tokenizer(num_words = num_words, split=' ')
tokenizer.fit_on_texts(sents)
seqs = tokenizer.texts_to_sequences(sents)
pad_seqs = pad_sequences(seqs, maxlen)


# ## Encoder Model

# In[12]:


encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
emb_layer = Embedding(num_words, embed_dim,input_length = maxlen, name='Body-Word-Embedding', mask_zero=False)
x = emb_layer(encoder_inputs)
state_h = Bidirectional(LSTM(128, activation='relu', name='Encoder-Last-LSTM'))(x)
encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)


# ## Decoder Model

# In[13]:


decoded = RepeatVector(maxlen)(seq2seq_encoder_out)
decoder_lstm = Bidirectional(LSTM(128, return_sequences=True, name='Decoder-LSTM-before'))
decoder_lstm_output = decoder_lstm(decoded)
decoder_dense = Dense(num_words, activation='softmax', name='Final-Output-Dense-before')
decoder_outputs = decoder_dense(decoder_lstm_output)


# ## Combining Model and Training

# In[14]:


seq2seq_Model = Model(encoder_inputs, decoder_outputs)
seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')
history = seq2seq_Model.fit(pad_seqs, np.expand_dims(pad_seqs, -1),
          batch_size=batch_size,
          epochs=10)


# In[17]:


vecs = encoder_model.predict(pad_seqs)


# In[16]:


sentence = "here's a sample unseen sentence"
seq = tokenizer.texts_to_sequences([sentence])
pad_seq = pad_sequences(seq, maxlen)
sentence_vec = encoder_model.predict(pad_seq)[0]


# ## References

# - [Building autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
# - [Training an AutoEncoder to Generate Text Embeddings](http://yaronvazana.com/2019/09/28/training-an-autoencoder-to-generate-text-embeddings/)
