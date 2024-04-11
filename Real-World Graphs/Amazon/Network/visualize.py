import networkx as nx
import matplotlib.pyplot as plt
import pickle


print("Loading Graph")

file = open("network_data.pkl", "rb")
G,Cls = pickle.load(file)
n = G.shape[0]

G = nx.to_networkx_graph(G)
G.remove_edges_from(nx.selfloop_edges(G))
communities = Cls[50]

print("Drawing")

colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]*5

# Compute positions for the node clusters as if they were themselves nodes in a
# supergraph using a larger scale factor
supergraph = nx.cycle_graph(len(communities))
#superpos = nx.spring_layout(G, scale=10, seed=429)
superpos = nx.circular_layout(supergraph, scale=20)

# Use the "supernode" positions as the center of each node cluster
centers = list(superpos.values())
pos = {}
for center, comm in zip(centers, communities):
    pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center))

# Nodes colored by cluster
for i,nodes in enumerate(communities):
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes, node_color=colors[i], node_size=5)
nx.draw_networkx_edges(G, pos=pos)

plt.tight_layout()
plt.show()