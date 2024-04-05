import sys
sys.path.insert(0, "../../")

from experiment_functions import *
import pickle

from joblib import Parallel, delayed 

# parameters
betas = [1,2]               # model degree
p = 0.1                     # treatment budget
qs = np.linspace(p,1,19)    # effective treatment budget
r = 10000                   # number of replications

print("Loading Graph")

graph_file = open("../Network/data.pkl", "rb")
G,Cl,membership = pickle.load(graph_file)

n = G.shape[0]             # graph size
h = homophily_effects(G)

data = { "q": [], "beta": [], "tte_hat": [], "est": [] }

def estimate_two_stage(fY,q,r,beta):
    Q = np.linspace(0, q, beta+1)
    Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,r)  # U is n x r
    tte_hat = pi_estimate_tte_two_stage(fY(Z),p/q,Q)
    e_tte_hat_given_u = q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1)

    return (q, tte_hat, e_tte_hat_given_u)

for beta in betas:
    fY = pom_ugander_yin(G,h,beta)
    TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
    print("beta: {}\t True TTE: {}".format(beta,TTE))

    for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,q,r,beta))(q) for q in qs):
        data["q"] += [q]*(2*r)
        data["beta"] += [beta]*(2*r)
        data["est"] += ["real"]*r
        data["tte_hat"] += list(TTE_hat - TTE)
        data["est"] += ["exp"]*r
        data["tte_hat"] += list(E_given_U - TTE)

file = open("pi_data.pkl", "wb")
pickle.dump((data), file)
file.close()