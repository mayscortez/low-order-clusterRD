# Main file for running experiments
from main_increasing_edges import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

beta = [1,2,3]
graphNum = 1     
T = 1
B = 0.06
p = 1
design = "bernoulli"
for b in beta:
    print('=====================================')
    print('Experiments for degree: {}'.format(b))
    print('=====================================')
    main(b, graphNum, T, B, p, design)