#!/usr/bin/env python
# coding: utf-8

# # Stanford Network Analysis Package

# In[1]:


import snap
from IPython.display import Image


# ## Data Types

# In[2]:


G = snap.LoadEdgeList(snap.PNGraph, "../../RepositoryData/data/cit-HepTh.txt")


# In[3]:


## Get node degrees
CntV = snap.TIntPrV()
snap.GetOutDegCnt(G, CntV)
for p in CntV:
    print("degree %d: count %d" % (p.GetVal1(), p.GetVal2()))


# In[5]:


print(snap.GetClustCf(G)) # clustering coefficient
print(snap.GetTriads(G))# diameter
print(snap.GetBfsFullDiam(G, 10))


# In[ ]:


## Betweenness centrality
Nodes = snap.TIntFltH()
Edges = snap.TIntPrFltH()
snap.GetBetweennessCentr(G, Nodes, Edges, 1.0)
# for node in Nodes:
#     print("node: %d centrality: %f" % (node, Nodes[node]))
# for edge in Edges:
#     print("edge: (%d, %d) centrality: %f" % (edge.GetVal1(), edge.GetVal2(), Edges[edge]))


# In[2]:


Graph = snap.GenRndGnm(snap.PNGraph, 10, 20)
Nodes = snap.TIntFltH()
Edges = snap.TIntPrFltH()
snap.GetBetweennessCentr(Graph, Nodes, Edges, 1.0)
for node in Nodes:
    print("node: %d centrality: %f" % (node, Nodes[node]))
for edge in Edges:
    print("edge: (%d, %d) centrality: %f" % (edge.GetVal1(), edge.GetVal2(), Edges[edge]))


# In[3]:


snap.PlotClustCf(Graph, "example", "Directed graph - clustering coefficient")
snap.DrawGViz(Graph, snap.gvlDot, "graph.png", "graph 1")
Image('graph.png')


# In[4]:


NIdColorH = snap.TIntStrH()
NIdColorH[0] = "green"
NIdColorH[1] = "red"
NIdColorH[2] = "purple"
NIdColorH[3] = "blue"
NIdColorH[4] = "yellow"
Network = snap.GenRndGnm(snap.PNEANet, 5, 10)
snap.DrawGViz(Network, snap.gvlSfdp, "network.png", "graph 3", True, NIdColorH)
Image('network.png')


# In[5]:


UGraph = snap.GenRndGnm(snap.PUNGraph, 10, 40)
snap.DrawGViz(UGraph, snap.gvlNeato, "graph_undirected.png", "graph 2", True)
Image('graph_undirected.png')


# - `snap.TintV()`
# - `snap.T

# ## References
# 
# - [SNAP](https://snap.stanford.edu/snap/index.html)
# - [SNAP Tutorial/Documentation](https://snap.stanford.edu/snappy/doc/index.html)
