# This file is to run the experiments where we vary the correlation between treatment effect type and community type
from main_extrapolation import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

beta = [1,2,3,4]  # for which polynomial model degrees do you want to run an experiment for?
budget = 0.06     # what is p, your treatment budget?
phis = [0]      # for which covariate balance levels do you want to run experiments for?
design = "bernoulli" # how do you want to choose clusters? options: "complete" or "bernoulli"
p_in = 0.4        # what is the edge probability within one cluster? (to keep expected degree at 10, should be a number between 0 and 0.5)
graphNum = 30     # how many graph models G to average over
U = 30            # how many cluster samples U to average over
T = 1             # how many treatment samples z to average over

for b in range(len(beta)):
    print('============================================================================================')
    print('Extrapolation experiments for degree: {} ({} design, p_in = {}, budget = {})'.format(beta[b], design, p_in, budget))
    print('============================================================================================')
    for phi in phis:
        print('phi = {}'.format(phi))
        print()
        main(beta[b], graphNum, T, U, budget, phi, p_in, design)