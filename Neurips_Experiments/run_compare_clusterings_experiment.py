import json
import pickle
import pymetis
import argparse

from experiment_functions import *
from itertools import product
from joblib import Parallel, delayed 

def estimate_two_stage(fY,Cl,n,p,q,r,beta):
    Q = np.linspace(0,q,beta+1)

    tte_hat = []
    e_tte_hat_given_u = []
    for _ in range(r//100):
        Z,U = staggered_rollout_two_stage(n,Cl,p,Q,100) 
        tte_hat = np.append(tte_hat,pi_estimate_tte_two_stage(fY(Z),p,Q))
        e_tte_hat_given_u = np.append(e_tte_hat_given_u, q/(n*p)*np.sum(fY(U) - fY(np.zeros(n)),axis=1))

    return (q, tte_hat, e_tte_hat_given_u)

def run_experiment(G,Cls,fixed,varied,r):
    n = G.shape[0]
    h = homophily_effects(G)

    # adjacency list representation
    A = [[] for _ in range(n)]
    for i,j in zip(*G.nonzero()):
        A[i].append(j)
        A[j].append(i)

    betas = [fixed["beta"]] if "beta" in fixed else varied["beta"] 
    ncs = [fixed["nc"]] if "nc" in fixed else varied["nc"]
    ps = [fixed["p"]] if "p" in fixed else varied["p"]

    data = { "q":[], "clustering":[], "bias":[], "var":[], "var_s":[] }
    if "beta" in varied: data["beta"] = []
    if "nc" in varied: data["nc"] = []
    if "p" in varied: data["p"] = []

    cluster_dict = {nc:{} for nc in ncs}

    for nc in ncs:
        print("Preparing Clusterings with {} Clusters".format(nc))

        cluster_dict[nc]["feature"] = Cls[nc]

        _,membership = pymetis.part_graph(nparts=nc,adjacency=A)
        membership = np.array(membership)
        Cl_graph = []
        for i in range(nc):
            Cl_graph.append(np.where(membership == i)[0])

        cluster_dict[nc]["graph"] = Cl_graph

        # randomly chosen balanced clustering
        membership = np.array(list(range(nc))*(n//nc+1))[:n]
        np.random.shuffle(membership)

        Cl_random = []
        for i in range(nc):
            Cl_random.append(np.where(membership == i)[0])

        cluster_dict[nc]["random"] = Cl_random

        cluster_dict[nc]["none"] = []

    for beta in betas:
        fY = pom_ugander_yin(G,h,beta)
        TTE = np.sum(fY(np.ones(n))-fY(np.zeros(n)))/n

        for nc,p in product(ncs,ps):
            for label,Cl in cluster_dict[nc].items():
                for (q,TTE_hat,E_given_U) in Parallel(n_jobs=-1, verbose=10)(delayed(lambda q : estimate_two_stage(fY,Cl,n,p,q,r,beta))(q) for q in np.linspace(p,1,16)):
                    data["q"].append(q)
                    data["clustering"].append(label)
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