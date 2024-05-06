import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


sns.set_theme()

print("Loading Graph")

file = open("data.pkl", "rb")
G,Cls = pickle.load(file)
n = G.shape[0]
Cl = Cls[42]

d = np.sum(G,axis=1)
l = np.array([len(c) for c in Cl])
print(l)

i = 20  # large 14 4 21

G = G.todense()
H = G[:,Cl[i]]

labels = []
for j in Cl[i]:
    labels.append(int(d[j] - np.sum(H[j,:])))

H = H[Cl[i],:]
m = H.shape[0]
H = H - np.eye(m)

Gr = nx.to_networkx_graph(H)
#isolates = list(nx.isolates(Gr))
labels = {i:l for (i,l) in enumerate(labels)}# if i not in isolates}
#Gr.remove_nodes_from(isolates)
#nx.spring_layout(Gr)
#fruchterman_reingold_layout(Gr)
nx.draw(Gr, pos=nx.spring_layout(Gr), node_color='r', node_size=240, with_labels=True, labels=labels)
plt.show()

# d = np.sum(G,axis=1)
# print(np.min(d))
# print(np.max(d))
# print(np.average(d))

# plt.hist(b,bins=range(0,250,2))
# plt.xlabel("In-Degree")
# plt.ylabel("Frequency")
# plt.subplots_adjust(bottom=0.25)
# plt.show()

exit()