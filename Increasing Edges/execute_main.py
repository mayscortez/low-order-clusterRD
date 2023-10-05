# Main file for running experiments
from main_increasing_edges import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

beta = [1,2,3]
graphNum = 50     
T = 50
for b in beta:
    main(b, graphNum, T)