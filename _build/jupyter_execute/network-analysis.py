# Network Analysis



from google.colab import drive
drive.mount('/content/drive')

## Word Similarities from Embeddings

If necessary, install `spacy` and the Chinese language model `zh_core_web_lg` (glove embeddings). 

!pip install spacy==2.3
!spacy download zh_core_web_lg
!pip install pyvis

Load the packages.

import spacy
nlp_zh = spacy.load('zh_core_web_lg')

near_syns = ['覺得','認為','宣稱','表示','強調','顯示', '說明','指出','提出','主張']


Inspect the word vectors matrix from the spacy model.

glove_word_vectors = nlp_zh.vocab.vectors
print('Spacy GloVe word vectors Shape: (vocab_size, embedding_dim)',glove_word_vectors.shape)
len(nlp_zh.vocab.vectors)

Check the similarities of 認為 with the other words in the near-syns

w1 = nlp_zh.vocab['認為']
w2 = nlp_zh.vocab['覺得']

# w1 similarities with others
for w in near_syns:
    if w !=w1:
        w_text = w
        w =nlp_zh.vocab[w]
        print(w_text, ':', w1.similarity(w))

To reduce the computation cost, extract the vocabulary of the Chinense model by excluding:
- ascii characters
- digits
- punctuations

And also, consider only two-character words.

import numpy as np
vocab = list(nlp_zh.vocab.strings)
#vocab = [w.text for w in nlp_zh.vocab if np.count_nonzero(w.vector) and not w.is_ascii and not w.is_punct]
# ]
print(len(vocab))
print(vocab[20000:20200])

target_word = '覺得'
word_sim = []
# check each word in vocab its simi with target_Word

target_word_vocab = nlp_zh.vocab[target_word]
for w in vocab:
    w_vocab = nlp_zh.vocab[w]
    if w_vocab.vector is not None and np.count_nonzero(w_vocab.vector):
        word_sim.append((w, target_word_vocab.similarity(w_vocab)))

sorted(word_sim, key=lambda x:x[1], reverse=True)[:10]

Each `vocab` has several properties defined in *spacy* that are useful for filtering irrelevant words before computing the word similarities

#w.is_lower == word.is_lower and w.prob >= -15

w1 = nlp_zh.vocab['覺得']
w2 = nlp_zh.vocab['ship']

print(w2.is_ascii)
print(w2.is_currency)
print(w2.is_punct)

Define functions to extract top-N similar words

- Functions taken from [this SO discussion thread](https://stackoverflow.com/questions/57697374/list-most-similar-words-in-spacy-in-pretrained-model)
- Deal with the computation efficiency problems (big matrices)

from numba import jit

@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


## Efficient version
def most_similar_v1(word, topn=5):
  word = nlp_zh.vocab[str(word)]
  queries = [
      w for w in nlp_zh.vocab 
      if np.count_nonzero(w.vector) and not w.is_ascii and not w.is_punct and len(w.text)==2
  ]

  #by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)

  by_similarity = sorted(queries, key=lambda w: cosine_similarity_numba(w.vector, word.vector), reverse=True)
    
    
  return [(w.text,w.similarity(word)) for w in by_similarity[:topn+1] if w.text != word.text]


## Naive version

def most_similar_v2(word, topn=5):
  word = nlp_zh.vocab[str(word)]
  queries = [
      w for w in nlp_zh.vocab 
      if np.count_nonzero(w.vector) and not w.is_ascii and not w.is_punct and len(w.text)==2
  ]

  by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
  #by_similarity = sorted(queries, key=lambda w: cosine_similarity_numba(w.vector, word.vector), reverse=True)

  return [(w.text,w.similarity(word)) for w in by_similarity[:topn+1] if w.text != word.text]



Test the time needed in different versions

%%time
most_similar_v1("覺得", topn=3)

%%time
most_similar_v2("覺得", topn=3)

## Defining Nodes for the Network

- Extract top N similar words for all near-syns
- These top N context words will form the basis for the nodes of the network

%%time
near_syn_topn = dict([(w, most_similar_v1(w, topn=1000)) for w in near_syns])

Top 10 similar words for each synonym in the list.

For example, the top 10 similar words for 覺得:

near_syn_topn[near_syns[0]][:10]

Convert the tuples into a list, which is easier to be imported into the graph structure.

near_syn_topn_list = []
for w, s in near_syn_topn.items():
    for s_w, s_s in s:
        near_syn_topn_list.append((w, s_w, s_s))

print(near_syn_topn_list[:10])
print(len(near_syn_topn_list))

import pandas as pd
df = pd.DataFrame(near_syn_topn_list,columns=['w1','w2','sim'])
df[df['sim']>0.6]

## Define Connections in-between Nodes

- While context nodes have already had connections (i.e., edges) to the key nodes (i.e., near-syns), these context nodes may themselves be inter-connected due to their semantic similarity
- We again utilize the `spacy` language model to determine their semantic similarities.
- These similarities serve as the basis for the edges of the network

We first identify all potential nodes for the network and then compute their pairwise similarities based on `spacy` Glove embeddings.

- `nodes_id`: include all the possible nodes of the graph.
- `edges_df`: include all the context-key and context-context edges of the graph.

WORD_SIMILARITY_CUTOFF = 0.65 # collexemes and target words
df2 = df[df['sim'] > WORD_SIMILARITY_CUTOFF]
nodes_id = list(set(list(df2['w2'].values) + list(df2['w1'].values)))
#print(nodes_id)
print(len(nodes_id))
# word vectors of all nodes
#x = np.array([nlp_zh(w).vector for w in nodes_id])

m = len(list(nodes_id))
distances = np.zeros((m,m))

for i in range(m):
    for j in range(m):  
        distances[i,j] = nlp_zh.vocab[nodes_id[i]].similarity(nlp_zh.vocab[nodes_id[j]])
# flatten        
EMBEDDING_CUTOFF = 0.75

#print(node_names)
distances_flat = []

for i in range(m):
    for j in range(m):
        if distances[i,j]> EMBEDDING_CUTOFF and i != j:
            distances_flat.append((nodes_id[i], nodes_id[j], distances[i,j]))

edges_df = pd.DataFrame(distances_flat, columns=['w1','w2','sim'])
print(edges_df.shape)

We then combine the context-key edges with the context-context edges. These edges are the final edges for the graph.

edges_df = edges_df.append(df2)
edges_df.loc[100:120,:]

## Creating a Network

- We use `networkx` to first create a graph and compute relevant node-level metrics, e.g., centralities.
- We then create two data frames for aesthetic specification of the graph:
  - `nodes_df`
  - `edges_df`
- We use `pyvis` to visualizae the network

import networkx as nx
from pyvis.network import Network
#import pyvis.options as options
#from sklearn.preprocessing import MinMaxScaler
#from scipy.spatial.distance import cosine
#G = nx.Graph()

## A function to rescale metrics for plotting
def myRescaler(x):
    x = np.array(x)
    y = np.interp(x, (x.min(), x.max()), (5, 20))
    return list(y)

Create `nodes_df`.

G= nx.from_pandas_edgelist(edges_df, 'w1','w2','sim')

nodes_df = pd.DataFrame({'id':list(G.nodes),
                         'betweenness': myRescaler(list(nx.betweenness_centrality(G).values())),
                         'eigenvector': myRescaler(list(nx.eigenvector_centrality(G).values()))})
nodes_df['size']=[5 if i not in near_syns else 10 for i in nodes_id]
nodes_df['size2']= [i if i not in near_syns else 30 for i in nodes_df['eigenvector']]
nodes_df['group'] = ['KEY' if nodes_df.loc[i,'id'] in near_syns else 'CONTEXT' for i in range(nodes_df.shape[0])]
nodes_df['color'] = ['lightpink' if nodes_df.loc[i,'group']=='KEY' else 'lightblue' for i in range(nodes_df.shape[0])]
nodes_df['borderWidthSelected'] = list(np.repeat(20.0, nodes_df.shape[0]))


## Visualizing a Network

Plotting the network using `pyvis`.

Gvis = Network("768px","1600px", notebook=False,heading="Semantic Network")
# # Gvis.from_nx(G)
edges_in = list(edges_df.to_records(index=False))
#Gvis.add_nodes(list(G.nodes), value=nodes_df['size2'], color=nodes_df['color'], borderWidthSelected = nodes_df['borderWidthSelected'])

for i in range(nodes_df.shape[0]):
  Gvis.add_node(list(G.nodes)[i], value=nodes_df.loc[i,'size2'], group=nodes_df.loc[i,'group'])#, color=nodes_df.loc[i,'color'], borderWidthSelected = nodes_df.loc[i,'borderWidthSelected'])

Gvis.add_edges(edges_in)
#Gvis.show_buttons()
Gvis.set_options("""
  var options = {
    "nodes": {
      "borderWidth": 0,
      "color": {
        "highlight": {
          "border": "rgba(221,171,197,1)",
          "background": "rgba(248,178,255,1)"
        }
      },
      "shadow": {
        "enabled": true
      }
    },
    "edges": {
      "color": {
        "highlight": "rgba(255,192,200,1)",
        "inherit": false
      },
      "smooth": false
    },
    "interaction": {
      "hover": true,
      "navigationButtons": true
    },
    "manipulation": {
      "enabled": true
    },
    "physics": {
      "barnesHut": {
        "springLength": 270
      },
      "minVelocity": 0.75
    }
  }
""")


  # groups: {
  #   myGroup: {color:{background:'red'}, borderWidth:3}
  # }



Gvis.show('/content/drive/My Drive/_MySyncDrive/Repository/python-notes/Gvis.html')

## References

- [`vis.js` Documentation](https://visjs.github.io/vis-network/docs/network/index.html)
