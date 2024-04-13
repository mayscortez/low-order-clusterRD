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
betas = [1,2]               # model degree
ncs = [50,100,150]          # number of clusters
p = 0.1                     # treatment budget
qs = np.linspace(p,1,16)    # effective treatment budget
r = 10000                   # number of replications

##############################################

data = { "q":[], "nc":[], "beta":[], "bias":[], "var":[], "var_s":[]}

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    for _ in range(r//1000):
        Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,1000)  # U is n x r
        tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p/q,Q))
        e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)

for nc in ncs:
    for beta in betas:
        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
        print("nc: {}\t beta: {}\t True TTE: {}".format(nc,beta,TTE))
        
        for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cls[nc],q,r,beta))(q) for q in qs):
            data["q"].append(q)
            data["beta"].append(beta)
            data["nc"].append(nc)

            mean = np.average(TTE_hat)
            variance = np.average((TTE_hat - mean)**2)
            s_mean = np.average(E_given_U)
            s_variance = np.average((E_given_U - s_mean)**2)

            data["bias"].append(mean - TTE)
            data["var"].append(variance)
            data["var_s"].append(s_variance)

file = open("pi_data.pkl", "wb")
pickle.dump(data, file)
file.close()