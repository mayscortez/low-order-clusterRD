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

print("Calculating Homophily Effects")

G = scipy.sparse.csr_matrix(G)
h = homophily_effects(G)
#h = homophily_effects(G)

print("Preparing Features")

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

# parameters
betas = [1,2]               # model degree
ncs = [4,8,12,16,20]        # number of clusters
p = 0.1                     # treatment budget
qs = np.linspace(p,1,19)    # effective treatment budget
r = 1000                    # number of replications

##############################################

data = { "q": [], "nc": [], "beta": [], "tte_hat": [], "type": [] }

def estimate_two_stage(q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(Z,Y,p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(Y(U) - Y(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for nc in ncs:
    print("Assigning Units to {} Clusters".format(nc))
    _,feature_membership = pymetis.part_graph(nparts=nc,xadj=xadj,adjncy=adjncy,vweights=vweights,eweights=eweights)

    unit_membership = np.zeros(n)

    for i in range(n):
        unit_membership[i] = feature_membership[random.choice(features[i])]

    Cl = []
    for i in range(int(np.max(feature_membership))+1):
        Cl.append(list(np.where(unit_membership == i)[0]))

    print("Cluster Sizes:",[len(cl) for cl in Cl])

    for beta in betas:
        Y = pom_ugander_yin(G,h,beta)
        TTE = np.sum(Y(np.ones(n))-Y(np.zeros(n)))/n
        print("beta: {}\t True TTE: {}".format(beta,TTE))
        
        for _ in range(r//1000):
            for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(q,1000,beta))(q) for q in qs):
                data["q"] += [q]*2000
                data["beta"] += [beta]*2000
                data["nc"] += [nc]*2000
                data["type"] += ["real"]*1000
                data["tte_hat"] += list(TTE_hat - TTE)
                data["type"] += ["exp"]*1000
                data["tte_hat"] += list(E_given_U - TTE)

df = pd.DataFrame(data)

colors = ['#0296fb', '#e20287']

g = sns.FacetGrid(df, row="beta", col="nc")
g.map_dataframe(sns.lineplot, x="q", y="tte_hat", hue="type", estimator="mean", errorbar="sd", palette=colors)
plt.legend()
plt.show()
