# Stanford Network Analysis Package

```{note}
For some reasons, I can't have the snap module properly imported on mac.
So, currently this notebook only runs on Google Colab.

```

!pip install snap-stanford

import snap
from IPython.display import Image

Graph = snap.GenRndGnm(snap.PNGraph, 10, 20)
snap.DrawGViz(Graph, snap.gvlDot, "graph.png", "graph 1")
Image('graph.png')

UGraph = snap.GenRndGnm(snap.PUNGraph, 10, 40)
snap.DrawGViz(UGraph, snap.gvlNeato, "graph_undirected.png", "graph 2", True)
Image('graph_undirected.png')



NIdColorH = snap.TIntStrH()
NIdColorH[0] = "green"
NIdColorH[1] = "red"
NIdColorH[2] = "purple"
NIdColorH[3] = "blue"
NIdColorH[4] = "yellow"
Network = snap.GenRndGnm(snap.PNEANet, 5, 10)
snap.DrawGViz(Network, snap.gvlSfdp, "network.png", "graph 3", True, NIdColorH)
Image('network.png')