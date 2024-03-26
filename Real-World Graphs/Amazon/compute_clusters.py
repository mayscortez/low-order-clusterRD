import pickle
import pymetis
import numpy as np
from joblib import Parallel, delayed 

graph_file = open("graph_data.pkl", "rb")
G,features = pickle.load(graph_file)
n = G.shape[0]

A = features @ features.T

xadj = [0]
adjncy = []
eweights = []
for i in range(n):
    for j in np.nonzero(A[i,:])[0]:
        adjncy.append(j)
        eweights.append(int(A[i,j]))
    xadj.append(len(adjncy))

Cls = {}
for (nc,(_,membership)) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda nc : (nc,pymetis.part_graph(nparts=nc,xadj=xadj,adjncy=adjncy,eweights=eweights)))(nc) for nc in [100,200,300,400,500]):
    membership = np.array(membership)
    Cl = []
    for i in range(nc):
        Cl.append(np.where(membership == i)[0])
    Cls[nc] = Cl

file = open("cluster_data.pkl", "wb")
pickle.dump((G,Cls), file)
file.close()