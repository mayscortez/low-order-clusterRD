import json
import pickle
import argparse

from experiment_functions import *
from itertools import product
from joblib import Parallel, delayed 

def estimate_two_stage(fY,G,Cl,n,p,q,r,beta,gamma):
    Q = np.linspace(0,q,beta+1)
    P = np.linspace(0,p,beta+1)
    Clb = [[i] for i in range(n)]

    tte_hat = {"pi_cluster":[],
               "pi_bernoulli":[],
               "dm_cluster":[],
               "dm_bernoulli":[],
               "dmt_cluster":[],
               "dmt_bernoulli":[],
               "ht_cluster":[],
               "ht_bernoulli":[],
               "hajek_cluster":[],
               "hajek_bernoulli":[]
               }
   
    for _ in range(r//1000):
        Zc,_ = staggered_rollout_two_stage(n,Cl,p,Q,1000) 
        Yc = fY(Zc)

        tte_hat["pi_cluster"] = np.append(tte_hat["pi_cluster"],pi_estimate_tte_two_stage(Yc,p,Q))
        tte_hat["dm_cluster"] = np.append(tte_hat["dm_cluster"],dm_estimate_tte(Zc[1:,:,:],Yc[1:,:,:]))
        tte_hat["dmt_cluster"] = np.append(tte_hat["dmt_cluster"],dm_threshold_estimate_tte(Zc[1:,:,:],Yc[1:,:,:],G,gamma))
        (ht_estimate,hajek_estimate) = ht_hajek_estimate_tte(Zc[-1,:,:],Yc[-1,:,:],G,Cl,p,Q[-1])
        tte_hat["ht_cluster"] = np.append(tte_hat["ht_cluster"],ht_estimate)
        tte_hat["hajek_cluster"] = np.append(tte_hat["hajek_cluster"],hajek_estimate)

        Zb,_ = staggered_rollout_two_stage(n,Clb,p,P,1000)
        Yb = fY(Zb)

        tte_hat["pi_bernoulli"] = np.append(tte_hat["pi_bernoulli"],pi_estimate_tte_two_stage(Yb,p,P))
        tte_hat["dm_bernoulli"] = np.append(tte_hat["dm_bernoulli"],dm_estimate_tte(Zb,Yb))
        tte_hat["dmt_bernoulli"] = np.append(tte_hat["dmt_bernoulli"],dm_threshold_estimate_tte(Zb,Yb,G,gamma))
        (ht_estimate,hajek_estimate) = ht_hajek_estimate_tte(Zb[-1,:,:],Yb[-1,:,:],G,Clb,p,P[-1])
        tte_hat["ht_bernoulli"] = np.append(tte_hat["ht_bernoulli"],ht_estimate)
        tte_hat["hajek_bernoulli"] = np.append(tte_hat["hajek_bernoulli"],hajek_estimate)

    return (p, tte_hat)

def run_experiment(G,Cls,fixed,varied,r,gamma):
    n = G.shape[0]
    h = homophily_effects(G)

    betas = [fixed["beta"]] if "beta" in fixed else varied["beta"] 
    ncs = [fixed["nc"]] if "nc" in fixed else varied["nc"]
    qs = [fixed["q"]] if "q" in fixed else varied["q"]

    data = { "p":[], "est":[], "treatment":[], "bias":[], "var":[] }
    if "beta" in varied: data["beta"] = []
    if "nc" in varied: data["nc"] = []
    if "q" in varied: data["q"] = []

    for beta in betas:
        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        for nc,q in product(ncs,qs):
            for (p,results) in Parallel(n_jobs=-1, verbose=10)(delayed(lambda p : estimate_two_stage(fY,G,Cls[nc],n,p,max(q,p),r,beta,gamma))(p) for p in np.linspace(0.1,0.5,24)):

                data["p"] += [p]*len(results)
                if "beta" in varied: data["beta"] += [beta]*len(results)
                if "nc" in varied: data["nc"] += [nc]*len(results)
                if "q" in varied: data["q"] += [q]*len(results)

                for label,TTE_hat in results.items():
                    est,treatment = label.split("_")
                    data["treatment"].append(treatment)
                    data["est"].append(est)

                    mean = np.average(TTE_hat)
                    variance = np.average((TTE_hat - mean)**2)

                    data["bias"].append(mean - TTE)
                    data["var"].append(variance)

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
    gamma = j["gamma"]

    data = run_experiment(G,Cls,fixed,varied,r,gamma)

    out_file = network_folder + "/Experiments/" + exp_name + ".pkl"
    print(f"Writing output to {out_file}")
    of = open(out_file,'wb')
    pickle.dump(data,of)
    of.close()
