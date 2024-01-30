# Main file for running experiments
from main_increasing_edges import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

beta = [1,2,3,4]
graphNum = 30     
T = 30
budget = 0.06
probs = [3/5]
design = "bernoulli"
for b in beta:
    print('=====================================')
    print('Experiments for degree: {}'.format(b))
    print('=====================================')
    for q in probs:
        print('q = {}\n'.format(q))
        main(b, graphNum, T, budget, q, design)