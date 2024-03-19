import pickle
import re
import numpy as np
import scipy.sparse

feature_dict = {} # dictionary mapping each product to list of categories

for category,products in enumerate(open("communities_5000.txt", "r")):
    for product in re.split("[\t\n]",products)[:-1]:
        product = int(product)
        if product not in feature_dict:
            feature_dict[product] = [category]
        else:
            feature_dict[product].append(category)

n = len(feature_dict)
id = { u:i for (i,u) in enumerate(feature_dict.keys())} # dictionary mapping products in these categories to a unique id

features = np.zeros((n,5000))
for i,fi in feature_dict.items():
    features[id[i],fi] = 1

G_dict = { i:[i] for i in range(n)}  # dictionary mapping each product to list of neighbors

for line in open("edges.txt", "r"):
    m = re.match("^([0-9]*)\t([0-9]*)$",line)
    u,v = m.group(1,2)
    u,v = int(u),int(v)

    if u not in feature_dict or v not in feature_dict: continue

    G_dict[id[u]].append(id[v])
    G_dict[id[v]].append(id[u])

G = scipy.sparse.lil_array((n,n))
    
for i in range(n):
    G[i,G_dict[i]] = 1

G = G.tocsr()

file = open("graph_data.pkl", "wb")
pickle.dump((G,features), file)
file.close()
