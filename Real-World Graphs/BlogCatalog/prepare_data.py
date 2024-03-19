import pickle
import re
import numpy as np
import scipy.sparse

n = 10312
G_dict = { i:[i] for i in range(n)}  # dictionary mapping each vertex to list of neighbors

for line in open("edges.csv", "r"):
    m = re.match("^([0-9]*),([0-9]*)$",line)
    u,v = m.group(1,2)
    u,v = int(u)-1,int(v)-1

    G_dict[u].append(v) 
    G_dict[v].append(u)

G = scipy.sparse.lil_array((n,n))
    
for i in range(n):
    G[i,G_dict[i]] = 1

G = G.tocsr()

feature_dict = { i:[] for i in range(n)} # dictionary mapping each vertex to list of communities

all_features = set()
for line in open("communities.csv", "r"):
    m = re.match("^([0-9]*),([0-9]*)$",line)
    v,c = m.group(1,2)
    v,c = int(v)-1,int(c)-1

    feature_dict[v].append(c)
    all_features.add(c)

nf = len(all_features)

features = np.zeros((n,nf))
for i,fi in feature_dict.items():
    features[i,fi] = 1

file = open("graph_data.pkl", "wb")
pickle.dump((G,features), file)
file.close()
