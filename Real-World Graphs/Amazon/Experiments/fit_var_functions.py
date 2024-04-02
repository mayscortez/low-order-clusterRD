import sys
sys.path.insert(0, "../../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle

print("Loading Graph")

file = open("../Network/data.pkl", "rb")
G,Cls = pickle.load(file)
n = G.shape[0]

print("Calculating Homophily Effects")

h = homophily_effects(G)

# parameters
beta = 2                    # model degree
nc = 100                    # number of clusters
p = 0.2                     # treatment budget
qs = np.linspace(p,1,16)    # effective treatment budget
r = 10000                   # number of replications

Cl = Cls[nc]

print("Calculating Potential Outcomes")
fY = pom_market(G,h,beta)

##############################################

data = { "q": [], "nc": [], "tte_hat": [], "est": [] }

print("Calculating Ls")

L = LPis(fY,Cl,n)
Lj = [np.sum(fY(e(n,[j])) - fY(np.zeros(n))) for j in range(n)]
Ljjp = {}
for i in range(100):
    C = Cl[i]
    print("Calculating effects for Cluster {}".format(i))
    for j in C:
        for jp in C:
            if jp == j: continue
            Ljjp[frozenset([j,jp])] = np.sum(fY(e(n,[j,jp])) - fY(np.zeros(n))) - Lj[j] - Lj[jp]

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(fY(Z),p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

print("nc: {}\t True TTE: {}".format(nc,beta,TTE))

for _ in range(r//1000):
    for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cls[nc],q,1000,beta))(q) for q in qs):
        data["q"] += [q]*2000
        data["nc"] += [nc]*2000
        data["est"] += ["real"]*1000
        data["tte_hat"] += list(TTE_hat - TTE)
        data["est"] += ["exp"]*1000
        data["tte_hat"] += list(E_given_U - TTE)

file = open("fit_poly.pkl", "wb")
pickle.dump((data,L,Lj,Ljjp,Cl), file)
file.close()