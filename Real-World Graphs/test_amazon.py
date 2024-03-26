from experiment_functions import *

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from joblib import Parallel, delayed 

print("Loading Graph")

file = open("Amazon/data.pkl", "rb")
G,Cls = pickle.load(file)
n = G.shape[0]

print("Calculating Homophily Effects")

h = homophily_effects(G)

# parameters
betas = [1,2]               # model degree
ncs = [100,300,500]         # number of clusters
p = 0.1                     # treatment budget
qs = np.linspace(p,1,19)    # effective treatment budget
r = 10000                   # number of replications

##############################################

data = { "q": [], "nc": [], "beta": [], "tte_hat": [], "est": [] }

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(fY(Z),p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for nc in ncs:
    for beta in betas:
        fY = pom_market(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
        print("nc: {}\t beta: {}\t True TTE: {}".format(nc,beta,TTE))
        
        for _ in range(r//1000):
            for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cls[nc],q,1000,beta))(q) for q in qs):
                data["q"] += [q]*2000
                data["beta"] += [beta]*2000
                data["nc"] += [nc]*2000
                data["est"] += ["real"]*1000
                data["tte_hat"] += list(TTE_hat - TTE)
                data["est"] += ["exp"]*1000
                data["tte_hat"] += list(E_given_U - TTE)

df = pd.DataFrame(data)

colors = ['#0296fb', '#e20287']

g = sns.FacetGrid(df, row="beta", col="nc")
g.map_dataframe(sns.lineplot, x="q", y="tte_hat", hue="est", estimator="mean", errorbar="sd", palette=colors)
plt.legend()
