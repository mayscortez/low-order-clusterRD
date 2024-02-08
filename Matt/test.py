from experiment_functions import *

### SBM Parameters
n = 200      # number of individuals
k = 20       # number of communities
pii = 0.5    # edge probability within community
pij = 0.02   # edge probability across communities

### other parameters
beta = 2     # model degree
p = 0.2      # treatment budget 
t = 5        # number of replications in experiment

G = sbm(n,k,pii,pij)

C = random_weights_degree_scaled(G,beta)
Y = outcomes(G,C,beta)
TTE = np.sum(Y(np.ones(n))-Y(np.zeros(n)))/n
print("True TTE: ",TTE)

alpha = true_poly(G,C,beta)
print("True Polynomial: {} p^2 + {} p + {}".format(alpha[2],alpha[1],alpha[0]))
F = lambda p: sum([alpha[k]*p**k for k in range(beta+1)])

P = (np.arange(10)*p/beta)[:beta+1]
 
Bias = 0
Var = 0
for _ in range(t):
    print("------------------------")
    Z = staggered_rollout_bern(n,P)

    for k in range(beta+1):
        print("p=",P[k],"True F(p)=",F(P[k]),"Estimated F(p)=",sum(Y(Z[k,:]))/n)

    TTE_est = pi_estimate_tte(Z,Y,P)
    Bias += TTE_est-TTE
    Var += (TTE_est-TTE)**2

print("------------------------")
print("Bias: ",Bias/t)
print("Variance: ",Var/t)




