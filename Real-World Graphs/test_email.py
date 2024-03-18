from experiment_functions import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import scipy

from joblib import Parallel, delayed 

# parameters
betas = [1,2]               # model degree
p = 0.1                     # treatment budget
qs = np.linspace(p,1,19)    # effective treatment budget
r = 10000                   # number of replications

print("Loading Graph")

graph_file = open("Email/graph_data.pkl", "rb")
G,Cl,membership = pickle.load(graph_file)
n = len(G)

# convert to adjacency matrix
A = np.zeros((n,n))
for i,Ni in enumerate(G):
    A[i,Ni] = 1
G = A

G = scipy.sparse.csr_matrix(G)
h = homophily_effects(G)

data = { "q": [], "beta": [], "tte_hat": [], "type": [] }

def estimate_two_stage(q,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(Z,Y,p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(Y(U) - Y(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for beta in betas:
    Y = pom_ugander_yin(G,h,beta)
    TTE = np.sum(Y(np.ones(n))-Y(np.zeros(n)))/n
    print("beta: {}\t True TTE: {}".format(beta,TTE))

    for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(q,beta))(q) for q in qs):
        data["q"] += [q]*(2*r)
        data["beta"] += [beta]*(2*r)
        data["type"] += ["real"]*r
        data["tte_hat"] += list(TTE_hat - TTE)
        data["type"] += ["exp"]*r
        data["tte_hat"] += list(E_given_U - TTE)

df = pd.DataFrame(data)

colors = ['#0296fb', '#e20287']

g = sns.FacetGrid(df, col="beta")
g.map_dataframe(sns.lineplot, x="q", y="tte_hat", hue="type", estimator="mean", errorbar="sd", palette=colors)
plt.legend()
plt.show()
