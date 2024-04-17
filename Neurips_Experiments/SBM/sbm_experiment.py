import sys
sys.path.insert(0, "../")

from experiment_functions import *
from joblib import Parallel, delayed 
import pickle

print("Constructing Graph")

n = 1000
k = 50
#edeg = 20
#piis = np.linspace(n/edeg,n/(k*edeg), 5)
pii = 0.5
pij = 0.01

c = n//k    # community size 
Cl = [list(range(c*i,c*(i+1))) for i in range(k)]

G = sbm(n,k,pii,pij)

# parameters
betas = [1,2,3]           # model degree
p = 0.2                   # treatment budget
qs = np.linspace(p,1,24)
r = 5000                  # number of replications

##############################################

data = { "q":[], "beta":[], "bias":[], "var":[], "var_s":[]}

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    for _ in range(r//1000):
        Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,1000)  # U is n x r
        tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p/q,Q))
        e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)

TTE_hat_dict = { beta:{ q:[] for q in qs} for beta in betas}
Bias_dict = { beta:{ q:[] for q in qs} for beta in betas}
E_given_U_dict = { beta:{ q:[] for q in qs} for beta in betas}

for _ in range():
    G = sbm(n,k,pii,pij)
    h = homophily_effects(G)

    for beta in betas:
        fY = pom_ugander_yin(G,h,beta)
        TTE_true = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=5)(delayed(lambda q : estimate_two_stage(fY,Cl,q,r,beta))(q) for q in qs):
            Bias_dict[beta][q].append(TTE_hat - TTE_true)
            TTE_hat_dict[beta][q].append(TTE_hat)
            E_given_U_dict[beta][q].append(E_given_U)

for beta in betas:
    for q in qs:
        data["q"].append(q)
        data["beta"].append(beta)
        data["bias"].append(np.average(Bias_dict[beta][q]))
        data["var"].append(np.average((TTE_hat_dict[beta][q] - np.average(TTE_hat_dict[beta][q]))**2))
        data["var_s"].append(np.average((E_given_U_dict[beta][q] - np.average(E_given_U_dict[beta][q]))**2))

file = open("sbm_data.pkl", "wb")
pickle.dump((data), file)
file.close()