import sys
sys.path.insert(0, "../../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle

print("Constructing Graph")

n = 1000
k = 50
pii = 0.5
pij = 0.05

G = sbm(n,k,pii,pij)

c = n//k    # community size 
Cl = [list(range(c*i,c*(i+1))) for i in range(k)]

print("Calculating Homophily Effects")

h = homophily_effects(G)

# parameters
betas = [1,2]                        # model degree
ps = [0.15,0.2,0.25,0.3,0.35]        # number of clusters
r = 10000                            # number of replications

##############################################

data = { "q": [], "p": [], "beta": [], "tte_hat": [], "est": [] }

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(fY(Z),p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for p in ps:
    for beta in betas:
        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
        print("p: {}\t beta: {}\t True TTE: {}".format(p,beta,TTE))

        qs = np.linspace(p,1,16)
        for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cl,q,r,beta))(q) for q in qs):
            data["q"] += [q]*2*r
            data["beta"] += [beta]*2*r
            data["p"] += [p]*2*r
            data["est"] += ["real"]*r
            data["tte_hat"] += list(TTE_hat - TTE)
            data["est"] += ["exp"]*r
            data["tte_hat"] += list(E_given_U - TTE)

file = open("sbm_data.pkl", "wb")
pickle.dump((data), file)
file.close()