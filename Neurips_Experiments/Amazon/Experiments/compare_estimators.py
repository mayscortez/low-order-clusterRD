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
ncs = [100,300,500]         # number of clusters
p = 0.1                     # treatment budget
qs = np.linspace(p,1,16)    # effective treatment budget
r = 1000                    # number of replications
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
    #tte_hat_ht = ht_estimate_tte(Z[1:,:,:],Y[1:,:,:],G,Cl,p,Q[1:])
    
    return (q, tte_hat_pi, tte_hat_dm, tte_hat_dmt)#, tte_hat_ht)

for nc in ncs:
    for beta in betas:
        fY = pom_market(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
        print("nc: {}\t beta: {}\t True TTE: {}".format(nc,beta,TTE))
        
        for _ in range(r//1000):
            for (q,TTE_pi,TTE_dm, TTE_dmt) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cls[nc],q,1000,beta))(q) for q in qs):
                data["q"] += [q]*3000
                data["beta"] += [beta]*3000
                data["nc"] += [nc]*3000
                data["est"] += ["PI"]*1000
                data["tte_hat"] += list(TTE_pi - TTE)
                data["est"] += ["DM"]*1000
                data["tte_hat"] += list(TTE_dm - TTE)
                data["est"] += ["DM(0.25)"]*1000
                data["tte_hat"] += list(TTE_dmt - TTE)
                # data["est"] += ["HT"]*1000
                # data["tte_hat"] += list(TTE_ht - TTE)

file = open("all_est_data.pkl", "wb")
pickle.dump((data), file)
file.close()