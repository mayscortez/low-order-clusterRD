import sys
sys.path.insert(0, "../../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle

print("Loading Graph")

file = open("../Network/data.pkl", "rb")
G,Cls = pickle.load(file)
n = G.shape[0]
h = homophily_effects(G)
fY = pom_ugander_yin(G,h,2)
TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
print("True TTE: {}".format(TTE))

# parameters
ncs = [50,200,500]             # number of clusters
ps = [0.1,0.2,0.3]             # treatment budget
r = 1000                       # number of replications

##############################################

data = { "q":[], "p":[], "nc":[], "bias":[], "var":[], "var_s":[] }

def estimate_two_stage(fY,Cl,p,q,r,beta=2):
    Q = np.linspace(0, q, beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    for _ in range(r//1000):
        Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,1000)  # U is n x r
        tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p/q,Q))
        e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)

for nc in ncs:
    Cl = Cls[nc]

    for p in ps:
        print("nc: {}\t p: {}\t".format(nc,p))

        for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cl,p,q,r))(q) for q in np.linspace(p,1,16)):
            data["q"].append(q)
            data["p"].append(p)
            data["nc"].append(nc)
            
            mean = np.average(TTE_hat)
            variance = np.average((TTE_hat - mean)**2)
            s_mean = np.average(E_given_U)
            s_variance = np.average((E_given_U - s_mean)**2)

            data["bias"].append(mean - TTE)
            data["var"].append(variance)
            data["var_s"].append(s_variance)

file = open("vary_p.pkl", "wb")
pickle.dump((data), file)
file.close()