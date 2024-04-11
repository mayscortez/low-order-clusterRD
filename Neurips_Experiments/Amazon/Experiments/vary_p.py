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
ncs = [50,200,500]             # number of clusters
ps = [0.1,0.2,0.3]             # treatment budget
r = 1000                       # number of replications

data = { "q": [], "p": [], "tte_hat": [], "est": [], "nc": [] }

def estimate_two_stage(fY,Cl,p,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(fY(Z),p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

fY = pom_ugander_yin(G,h,2)
TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
print("True TTE: {}".format(TTE))

for nc in ncs:
    Cl = Cls[nc]

    for p in ps:
        print("nc: {}\t p: {}\t".format(nc,p))

        for _ in range(r//1000):
            for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cl,p,q,1000,2))(q) for q in np.linspace(p,1,16)):
                data["q"] += [q]*2000
                data["p"] += [p]*2000
                data["nc"] += [nc]*2000
                data["est"] += ["real"]*1000
                data["tte_hat"] += list(TTE_hat - TTE)
                data["est"] += ["exp"]*1000
                data["tte_hat"] += list(E_given_U - TTE)

file = open("vary_p.pkl", "wb")
pickle.dump((data), file)
file.close()