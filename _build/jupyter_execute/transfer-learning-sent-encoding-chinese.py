# Universal Sentence Encoding

- Google released this pre-trained Universal Sentence Encoder, which supports 16 languages, including traditional Chinese!!
- [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3)
- This universal encoding allows us to compute the semantic similarities between sentences in one language as well as sentences across different languages



!pip3 install tensorflow_text>=2.0.0rc0

import tensorflow_hub as hub
import numpy as np
import tensorflow_text


# Some texts of different lengths.
chinese_sentences = ["今天天氣還不錯",
                     "我昨天去那家店買本書",
                     "他的名字是奶奶取的",
                     "這天氣也太美妙了"]

english_sentences = ["It's nice today",
                     "I bought a book at the store yesterday",
                     "His granny gave him this beautiful name",
                     "The weather is just lovely"]

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Compute embeddings.

ch_result = embed(chinese_sentences)
en_result = embed(english_sentences)
# Compute similarity matrix. Higher score indicates greater similarity.
similarity_matrix_ch = np.inner(ch_result, ch_result)
similarity_matrix_en = np.inner(en_result, en_result)
similarity_matrix_ce = np.inner(ch_result, en_result)

print(similarity_matrix_ch)

print(similarity_matrix_en)

print(similarity_matrix_ce)