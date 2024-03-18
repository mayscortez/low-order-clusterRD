from experiment_functions import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import random

import pymetis
from joblib import Parallel, delayed 
from itertools import combinations

print("Loading Graph")

graph_file = open("BlogCatalog/graph_data.pkl", "rb")
G,features = pickle.load(graph_file)
n = len(G)

# convert to adjacency matrix
A = np.zeros((n,n))
for i,Ni in enumerate(G):
    A[i,Ni] = 1
G = A

print("Clustering Features")

feature_dict = {}

for i,fi in features.items():
    for f in fi:
        if f not in feature_dict:
            feature_dict[f] = { "weight": 1, "neighbors": {}}
        else:
            feature_dict[f]["weight"] += 1
        
    for (f1,f2) in combinations(fi,2):
        if f2 not in feature_dict[f1]["neighbors"]:
            feature_dict[f1]["neighbors"][f2] = 1
            feature_dict[f2]["neighbors"][f1] = 1
        else:
            feature_dict[f1]["neighbors"][f2] += 1
            feature_dict[f2]["neighbors"][f1] += 1

xadj = [0]
adjncy = []
vweights = []
eweights = []

for _,f in feature_dict.items():
    vweights.append(f["weight"])
    for g,w in f["neighbors"].items():
        adjncy.append(g)
        eweights.append(w)
    xadj.append(len(adjncy))

_,feature_membership = pymetis.part_graph(nparts=6,xadj=xadj,adjncy=adjncy,vweights=vweights,eweights=eweights)

print("Assigning Units to Clusters")

unit_membership = np.zeros(n)

for i in range(n):
    unit_membership[i] = feature_membership[random.choice(features[i])]

Cl = []
for i in range(int(np.max(feature_membership))+1):
    Cl.append(list(np.where(unit_membership == i)[0]))

print("Cluster Sizes:",[len(cl) for cl in Cl])

print("Calculating Homophily Effects")
h = homophily_effects(G)

##############################################

# parameters
betas = [1,2]               # model degree
p = 0.1                     # treatment budget
qs = np.linspace(p,1,19)    # effective treatment budget
bs = np.linspace(0,1,2)     # magnitude of homophily
r = 10000                   # number of replications

data = { "q": [], "b": [], "beta": [], "tte_hat": [], "type": [] }

def estimate_two_stage(q,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(Z,Y,p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(Y(U) - Y(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for beta in betas:
    for b in bs:
        Y = pom_ugander_yin(G,b*h,beta)
        TTE = np.sum(Y(np.ones(n))-Y(np.zeros(n)))/n
        print("b: {}\t True TTE: {}".format(b,TTE))

        for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(q,beta))(q) for q in qs):
            data["q"] += [q]*(2*r)
            data["b"] += [b]*(2*r)
            data["beta"] += [beta]*(2*r)
            data["type"] += ["real"]*r
            data["tte_hat"] += list(TTE_hat - TTE)
            data["type"] += ["exp"]*r
            data["tte_hat"] += list(E_given_U - TTE)

df = pd.DataFrame(data)

colors = ['#0296fb', '#e20287']

g = sns.FacetGrid(df, row="beta", col="b")
g.map_dataframe(sns.lineplot, x="q", y="tte_hat", hue="type", estimator="mean", errorbar="sd", palette=colors)
plt.legend()
plt.show()
