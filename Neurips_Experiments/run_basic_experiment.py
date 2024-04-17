import json
import pickle
import argparse

from experiment_functions import *
from itertools import product
from joblib import Parallel, delayed 

def estimate_two_stage(fY,Cl,n,p,q,r,beta):
    Q = np.linspace(0,q,beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    for _ in range(r//1000):
        Z,U = staggered_rollout_two_stage(n,Cl,p/q,Q,1000) 
        tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p/q,Q))
        e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)

def run_experiment(G,Cls,fixed,varied,r):
    n = G.shape[0]
    h = homophily_effects(G)

    betas = [fixed["beta"]] if "beta" in fixed else varied["beta"] 
    ncs = [fixed["nc"]] if "nc" in fixed else varied["nc"]
    ps = [fixed["p"]] if "p" in fixed else varied["p"]

    data = { "q":[], "bias":[], "var":[], "var_s":[] }
    if "beta" in varied: data["beta"] = []
    if "nc" in varied: data["nc"] = []
    if "p" in varied: data["p"] = []

    for beta in betas:
        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        for nc,p in product(ncs,ps):
            for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=10)(delayed(lambda q : estimate_two_stage(fY,Cls[nc],n,p,q,r,beta))(q) for q in np.linspace(p,1,24)):
                data["q"].append(q)
                if "beta" in varied: data["beta"].append(beta)
                if "nc" in varied: data["nc"].append(nc)
                if "p" in varied: data["p"].append(p)

                mean = np.average(TTE_hat)
                variance = np.average((TTE_hat - mean)**2)
                s_mean = np.average(E_given_U)
                s_variance = np.average((E_given_U - s_mean)**2)

                data["bias"].append(mean - TTE)
                data["var"].append(variance)
                data["var_s"].append(s_variance)

    return data

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('jsonfile')
    args = p.parse_args()

    jf = open(args.jsonfile,'rb')
    j = json.load(jf)
    jf.close()

    exp_name = j["name"]
    network_folder = j["network"]
    in_file = j["input"]

    print("Loading Graph")

    nf = open(network_folder + "/" + in_file,'rb')
    G,Cls = pickle.load(nf)
    nf.close()

    fixed = j["fix"]
    varied = j["vary"]
    r = j["replications"]

    data = run_experiment(G,Cls,fixed,varied,r)

    out_file = network_folder + "/Experiments/" + exp_name + ".pkl"
    print(f"Writing output to {out_file}")
    of = open(out_file,'wb')
    pickle.dump(data,of)
    of.close()
