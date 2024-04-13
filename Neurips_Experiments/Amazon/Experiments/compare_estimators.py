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

data = { "q":[], "nc":[], "beta":[], "est": [], "bias":[], "sd":[]}

def estimate_two_stage(fY,Cl,q,r,beta):
    Q = np.linspace(0, q, beta+1)

    TTE_hat = {"PI":[], "DM":[], "DM({})".format(gamma):[], "HT":[], "Hajek":[]}

    for _ in range(r//1000):
        Z,_ = staggered_rollout_two_stage(n,Cl,p/q,Q,1000)  # U is n x r
        Y = fY(Z)
        TTE_hat["PI"] = np.append(TTE_hat["PI"],pi_estimate_tte_two_stage(Y,p/q,Q))
        TTE_hat["DM"] = np.append(TTE_hat["DM"],dm_estimate_tte(Z,Y))
        TTE_hat["DM({})".format(gamma)] = np.append(TTE_hat["DM({})".format(gamma)],dm_threshold_estimate_tte(Z,Y,G,gamma))
        TTE_hat["HT"] = np.append(TTE_hat["HT"],ht_estimate_tte(Z[1:,:,:],Y[1:,:,:],G,Cl,p,Q[1:]))
        TTE_hat["Hajek"] = np.append(TTE_hat["Hajek"],hajek_estimate_tte(Z[1:,:,:],Y[1:,:,:],G,Cl,p,Q[1:]))
    
    return (q, TTE_hat)

for nc in ncs:
    for beta in betas:
        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n
        print("nc: {}\t beta: {}\t True TTE: {}".format(nc,beta,TTE))
        
        for (q,TTE_hat) in Parallel(n_jobs=-1, verbose=20)(delayed(lambda q : estimate_two_stage(fY,Cls[nc],q,r,beta))(q) for q in qs):
            data["q"] += [q]*len(TTE_hat)
            data["beta"] += [beta]*len(TTE_hat)
            data["nc"] += [nc]*len(TTE_hat)

            for estimator,estimate in TTE_hat.items():
                data["est"].append(estimator)
                mean = np.average(estimate)
                data["bias"].append(mean - TTE)
                data["sd"].append(np.average((estimate - mean)**2)**(1/2))

file = open("compare_estimators.pkl", "wb")
pickle.dump(data, file)
file.close()