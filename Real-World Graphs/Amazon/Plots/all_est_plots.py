import seaborn.objects as so
import pandas as pd
import pickle

file = open('../Experiments/all_est_data.pkl', 'rb')
data = pickle.load(file)
file.close()

df = pd.DataFrame(data)

p = (
    so.Plot(df, x='q', y='tte_hat', color='est')
    .facet(row='beta', col='nc')
    .add(so.Line(), so.Agg())                 # line plot of expected bias
    .add(so.Band(), so.Est(errorbar='sd'))    # shading for standard deviation
    .layout(size=(10,8))
)

p.show()
p.save("all_est.png")
