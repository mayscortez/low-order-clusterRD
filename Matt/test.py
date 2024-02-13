from experiment_functions import *

### SBM Parameters
n = 200      # number of individuals
k = 20       # number of communities
pii = 0.5    # edge probability within community
pij = 0.02   # edge probability across communities

### other parameters
beta = 3     # model degree
p = 0.2      # treatment budget 
r = 1000     # number of replications per graph
R = 10       # number of graphs

Bias_s = 0
Var_s = 0
Bias_u = 0
Var_u = 0

for _ in range(R):
    G = sbm(n,k,pii,pij)

    C = random_weights_degree_scaled(G,beta)
    Y = outcomes(G,C,beta)
    TTE = np.sum(Y(np.ones(n))-Y(np.zeros(n)))/n
    #print("\nTrue TTE: ",TTE)

    P = (np.arange(10)*p/beta)[:beta+1]
    Z = staggered_rollout_bern(n,P,r)
    Zu = uncorrelated_bern(n,P,r)

    ########### True vs. Estimated Polynomial ###########
    # alpha = true_poly(G,C,beta)
    # print("True Polynomial: {} p^2 + {} p + {}".format(alpha[2],alpha[1],alpha[0]))
    # F = lambda p: sum([alpha[k]*p**k for k in range(beta+1)])

    # for k in range(beta+1):
    #     print("p=",P[k],"True F(p)=",F(P[k]),"Estimated F(p)=",sum(Y(Z[k,:,0]))/n)
    #####################################################

    TTE_ests_s = pi_estimate_tte(Z,Y,P)
    Bias_s += sum(TTE_ests_s-TTE)/r
    Var_s += sum((TTE_ests_s-TTE)**2)/r

    TTE_ests_u = pi_estimate_tte(Zu,Y,P)
    Bias_u += sum(TTE_ests_u-TTE)/r
    Var_u += sum((TTE_ests_u-TTE)**2)/r

print("------------------------")
print("Staggered Rollout: \tBias: {} \tVar: {}".format(Bias_s/R,Var_s/R))
print("Uncorrelated Rounds: \tBias: {} \tVar: {}".format(Bias_u/R,Var_u/R))





