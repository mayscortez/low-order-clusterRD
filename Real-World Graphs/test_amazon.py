from experiment_functions import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from kmodes.kmodes import KModes
from joblib import Parallel, delayed 
import scipy.sparse

print("Loading Graph")

graph_file = open("Amazon/graph_data.pkl", "rb")
G,features = pickle.load(graph_file)
n = G.shape[0]

print("Calculating Homophily Effects")

h = homophily_effects(G)

# parameters
betas = [1,2]               # model degree
ncs = [100]                 # number of clusters
p = 0.1                     # treatment budget
qs = np.linspace(p,1,19)    # effective treatment budget
r = 1000                    # number of replications

##############################################

data = { "q": [], "nc": [], "beta": [], "tte_hat": [], "type": [] }

def estimate_two_stage(Y,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(Z,Y,p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(Y(U) - Y(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for nc in ncs:
    print("Assigning Units to {} Clusters".format(nc))

    membership = KModes(n_clusters=nc, init="random", n_init=10, verbose=10, n_jobs=-1).fit_predict(features)

    Cl = []
    for i in range(nc):
        Cl.append(list(np.where(membership == i)[0]))

    print("Cluster Sizes:",[len(cl) for cl in Cl])

    for beta in betas:
        Y = pom_ugander_yin(G,h,beta)
        TTE = np.sum(Y(np.ones(n))-Y(np.zeros(n)))/n
        print("beta: {}\t True TTE: {}".format(beta,TTE))
        
        for _ in range(r//100):
            for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(Y,q,100,beta))(q) for q in qs):
                data["q"] += [q]*200
                data["beta"] += [beta]*200
                data["nc"] += [nc]*200
                data["type"] += ["real"]*100
                data["tte_hat"] += list(TTE_hat - TTE)
                data["type"] += ["exp"]*100
                data["tte_hat"] += list(E_given_U - TTE)

df = pd.DataFrame(data)

colors = ['#0296fb', '#e20287']

g = sns.FacetGrid(df, row="beta", col="nc")
g.map_dataframe(sns.lineplot, x="q", y="tte_hat", hue="type", estimator="mean", errorbar="sd", palette=colors)
plt.legend()
plt.show()
