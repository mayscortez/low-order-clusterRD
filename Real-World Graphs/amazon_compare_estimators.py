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
gamma = 0.25                # DM threshold

##############################################

data = { "q": [], "nc": [], "beta": [], "tte_hat": [], "est": [] }

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,_ = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    Y = fY(Z)
    tte_hat_pi = pi_estimate_tte_two_stage(Y,p/q,Q)
    tte_hat_dm = dm_estimate_tte(Z,Y)
    tte_hat_dmt = dm_threshold_estimate_tte(Z,Y,G,gamma)
    tte_hat_ht = ht_estimate_tte(Z,Y,G,Cl,p/q,Q)

    return (q, tte_hat_pi, tte_hat_dm, tte_hat_dmt, tte_hat_ht)

for nc in ncs:
    for beta in betas:
        fY = pom_market(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
        print("nc: {}\t beta: {}\t True TTE: {}".format(nc,beta,TTE))
        
        for _ in range(r//1000):
            for (q,TTE_pi,TTE_dm, TTE_dmt, TTE_ht) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cls[nc],q,1000,beta))(q) for q in qs):
                data["q"] += [q]*4000
                data["beta"] += [beta]*4000
                data["nc"] += [nc]*4000
                data["est"] += ["PI"]*1000
                data["tte_hat"] += list(TTE_pi - TTE)
                data["est"] += ["DM"]*1000
                data["tte_hat"] += list(TTE_dm - TTE)
                data["est"] += ["DM(0.25)"]*1000
                data["tte_hat"] += list(TTE_dmt - TTE)
                data["est"] += ["HT"]*1000
                data["tte_hat"] += list(TTE_ht - TTE)

df = pd.DataFrame(data)

#colors = ['#0296fb', '#e20287']

g = sns.FacetGrid(df, row="beta", col="nc")
g.map_dataframe(sns.lineplot, x="q", y="tte_hat", hue="est", estimator="mean", errorbar="sd") #palette=colors)
plt.legend()
plt.show()
