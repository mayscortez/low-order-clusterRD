import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import experiment_functions as ex

rng = np.random.RandomState(19025)

# testing shapes for _outcomes_dyadic
beta = 2
r = 10
n = 6
p = 0.5

alpha = np.arange(n)
deg1 = rng.rand(n,n)
deg2 = rng.rand(n,n)

Z = (rng.rand(beta+1,r,n) < p) + 0
Y = ex._outcomes_dyadic(Z,alpha,deg1,deg2)
print(Y.shape==Z.shape)

# testing shapes for pom_dyadic
G = ex.er(n,0.5)
A = G.toarray()
print(A.shape, type(A))

Z = (rng.rand(beta+1,r,n) < p) + 0
print(Z.shape)

params_unif = {'dist': 'uniform', 'direct': 1, 'indirect_deg1': 0.5, 'indirect_deg2': 0.5}
fy = ex.pom_dyadic(A,params_unif)

Y = fy(Z)
print(Y.shape==Z.shape)

params_bern = {'dist': 'bernoulli', 'direct': 0.5, 'indirect_deg1': 0.25, 'indirect_deg2': 0.25}
fy = ex.pom_dyadic(A,params_bern)

Y = fy(Z)
print(Y.shape==Z.shape)

# testing we can compute TTE with convenience function
TTE = ex.TTE(fy)
print(TTE)