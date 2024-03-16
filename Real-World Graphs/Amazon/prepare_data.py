import pickle
import re
import numpy as np

G_dict = {} # dictionary mapping each vertex to list of neighbors

products = set()

for line in open("edges.txt", "r"):
    m = re.match("^([0-9]*)\t([0-9]*)$",line)
    u,v = m.group(1,2)
    u,v = int(u),int(v)

    products.add(u)
    products.add(v)

    if u not in G_dict:
        G_dict[u] = [v]
    else:
        G_dict[u].append(v)

    if v not in G_dict:
        G_dict[v] = [u]
    else:
        G_dict[v].append(u)

features = { u:[] for u in products} # dictionary mapping each product to list of categories

for j,line in enumerate(open("communities_5000.txt", "r")):
    for u in re.split("[\t\n]",line)[:-1]:
        features[int(u)].append(j)

deleted = set()
for u in products:
    if len(features[u]) == 0:
        for v in G_dict[u]:
            G_dict[v].remove(u)
        del G_dict[u]
        del features[u]
        deleted.add(u)

products = products.difference(deleted)

id = { u:i for (i,u) in enumerate(products)}
G = []

for u in products:
    G_dict[u].append(u)
    G.append(sorted([id[v] for v in G_dict[u]]))

l = [len(v) for k,v in features.items()]

file = open("graph_data.pkl", "wb")
pickle.dump((G,features), file)
file.close()
