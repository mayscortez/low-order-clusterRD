import seaborn.objects as so
import pandas as pd
import pickle

file = open('../Experiments/pi_data.pkl', 'rb')
data = pickle.load(file)
file.close()

df = pd.DataFrame(data)

colors = ['#0296fb', '#e20287']

p = (
    so.Plot(df, x='q', y='tte_hat', color='est')
    .facet(col='beta')
    .scale(color=colors)
    .add(so.Line(), so.Agg())                 # line plot of expected bias
    .add(so.Band(), so.Est(errorbar='sd'))    # shading for standard deviation
    .layout(size=(8,4))
)

p.show()
p.save("bias_var.png")