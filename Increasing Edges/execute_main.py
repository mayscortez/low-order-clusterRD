# Main file for running experiments
from main_increasing_edges import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

beta = [2,3]
graphNum = 30     
T = 30
for b in beta:
    print('=====================================')
    print('Experiments for degree: {}'.format(b))
    print('=====================================')
    main(b, graphNum, T)