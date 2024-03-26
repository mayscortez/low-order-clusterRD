import re
import pickle
import pymetis
import numpy as np
import scipy.sparse
from joblib import Parallel, delayed 

print("Importing Product Features")

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

print("Importing Product Edges")

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

print("Constructing Feature Graph")

A = features @ features.T

xadj = [0]
adjncy = []
eweights = []
for i in range(n):
    for j in np.nonzero(A[i,:])[0]:
        adjncy.append(j)
        eweights.append(int(A[i,j]))
    xadj.append(len(adjncy))

print("Computing Clusterings")
Cls = {}
for (nc,(_,membership)) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda nc : (nc,pymetis.part_graph(nparts=nc,xadj=xadj,adjncy=adjncy,eweights=eweights)))(nc) for nc in range(50,1001,50)):
    membership = np.array(membership)
    Cl = []
    for i in range(nc):
        Cl.append(np.where(membership == i)[0])
    Cls[nc] = Cl

file = open("network_data.pkl", "wb")
pickle.dump((G,Cls), file)
file.close()