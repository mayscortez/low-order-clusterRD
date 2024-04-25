import sys
sys.path.insert(0, "../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle

print("Constructing Graph")

n = 1000
k = 20
c = n//k    # community size 
Cl = [list(range(c*i,c*(i+1))) for i in range(k)]

edeg = 20
#piis = np.linspace((edeg-1)/(n-1),(edeg-1)/(c-1),10)
num_p = 5
pmin = (edeg-1)/(n-1)
pmax = (edeg-1)/(c-1)
piis = [pmax - (num_p-i)**2/num_p**2 * (pmax-pmin) for i in range(num_p)]

# parameters
beta = 2                  # model degree
p = 0.2                   # treatment budget
qs = np.linspace(p,1,24)
r = 1000                  # number of replications

##############################################

data = { "q":[], "pii":[], "bias":[], "var":[], "var_s":[]}

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    #for _ in range(r//1000):
    Z,U = staggered_rollout_two_stage(n,Cl,p,Q,r) 
    tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p,Q))
    e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)

for pii in piis:
    pij = (edeg - pii*(c-1) - 1)/(n-c)

    TTE_hat_dict = {q:[] for q in qs}
    Bias_dict = {q:[] for q in qs}
    E_given_U_dict = {q:[] for q in qs}

    for _ in range(10):
        G = sbm(n,k,pii,pij)
        h = homophily_effects(G)

        fY = pom_ugander_yin(G,h,beta)
        TTE_true = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=5)(delayed(lambda q : estimate_two_stage(fY,Cl,q,r,beta))(q) for q in qs):
            Bias_dict[q].append(TTE_hat - TTE_true)
            TTE_hat_dict[q].append(TTE_hat)
            E_given_U_dict[q].append(E_given_U)

    for q in qs:
        data["pii"].append(pii)
        data["q"].append(q)
        data["bias"].append(np.average(Bias_dict[q]))
        data["var"].append(np.average((TTE_hat_dict[q] - np.average(TTE_hat_dict[q]))**2))
        data["var_s"].append(np.average((E_given_U_dict[q] - np.average(E_given_U_dict[q]))**2))

file = open("sbm_data_vary_connectivity.pkl", "wb")
pickle.dump((data), file)
file.close()