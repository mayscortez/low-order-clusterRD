import sys
sys.path.insert(0, "../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle

print("Constructing Graph")

n = 1000
k = 50
pii = 1/2
pij = 1/95

c = n//k    # community size 
Cl = [list(range(c*i,c*(i+1))) for i in range(k)]

# parameters
beta = 2        # model degree
p = 0.2         # treatment budget
r = 10000       # number of replications
q = 0.33    

bs = [1,9,17]

##############################################

data = { "q": [], "b": [], "tte_hat": [], "est": [] }

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(fY(Z),p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for _ in range(10):
    G = sbm(n,k,pii,pij)
    h = homophily_effects(G)
    
    for b in bs:
        fY = pom_ugander_yin(G,h*b,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
        print("q: {}\t pii: {}\t True TTE: {}".format(q,pii,TTE))

        qs = np.linspace(p,1,16)
        for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cl,q,r,beta))(q) for q in qs):
            data["q"] += [q]*2*r
            data["b"] += [b]*2*r
            data["est"] += ["real"]*r
            data["tte_hat"] += list(TTE_hat - TTE)
            data["est"] += ["exp"]*r
            data["tte_hat"] += list(E_given_U - TTE)

file = open("sbm_data_homophily.pkl", "wb")
pickle.dump((data), file)
file.close()