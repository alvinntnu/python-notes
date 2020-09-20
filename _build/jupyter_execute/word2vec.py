# Word2Vec

## Training 

from gensim.models import word2vec

%%time

# sentences = word2vec.Text8Corpus('../data/text8') 
# model = word2vec.Word2Vec(sentences, size=200, hs=1)

# ## It takes a few minutes to train the model (about 7min on my Mac)
model = word2vec.Word2Vec.load("../data/text8_model") ## load the pretrained model
print(model)

# with open('../data/text8', 'r') as f:
#     for i in range(2):
#         l = f.readline()
#         print(l)

# f = open('../data/text8', 'r')
# l=f.readline()
# print(l)
# f.close()

## Functionality of Word2Vec

model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)[0]

model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])

model.wv['computer']

## Save Model

#model.save("../data/text8_model")

## Load model
#model = word2vec.Word2Vec.load("../data/text8_model")

## identify the most that is the most semantically distant from the others from a word list
model.wv.doesnt_match("breakfast cereal dinner lunch".split())

model.wv.similarity('woman', 'man')

model.wv.similarity('woman', 'cereal')

 
model.wv.distance('man', 'woman')

## Save model keyed vector
#word_vectors = model.wv
#del model

## Evaluating Word Vectors

import os, gensim
module_path = gensim.__path__[0]
#print(module_path)
print(os.path.join(module_path, 'test/test_data','wordsim353.tsv'))
model.wv.evaluate_word_pairs(os.path.join(gensim.__path__[0], 'test/test_data','wordsim353.tsv'))

model.wv.accuracy(os.path.join(module_path, 'test/test_data', 'questions-words.txt'))[1]

## Loading Pre-trained Model

# from gensim.models import KeyedVectors
# load the google word2vec model
# filename = 'GoogleNews-vectors-negative300.bin'
# model = KeyedVectors.load_word2vec_format(filename, binary=True)

