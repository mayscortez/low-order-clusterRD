# This file is to run the experiments where we vary the correlation between treatment effect type and community type
from main_correlation import main
import os

path = os.getcwd()
print("Path = {}".format(path))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

beta = [1,2,3]
graphNum = 50    
T = 50
for b in beta:
    main(b, graphNum, T)