import seaborn.objects as so
import pandas as pd
import pickle

file = open('../Experiments/pi_data.pkl', 'rb')
data = pickle.load(file)
file.close()

df = pd.DataFrame(data)

def aggregate_stats(x):
    d = {}
    d['bias^2'] = x['tte_hat'].mean()**2
    d['variance'] = x['tte_hat'].var()
    return pd.Series(d, index=['bias^2', 'variance'])

df = (
    df.groupby(['q','nc','beta','est'])                              # consider each set of parameters separately
    .apply(aggregate_stats,include_groups=False)                     # calculate bias and variance
    .reset_index()
    .pivot(index=['q','nc','beta'], columns=['est'])                 # columns for overall / conditional variance
    .pipe(lambda s: s.set_axis(s.columns.map('_'.join), axis=1))     # fix column naming
    .reset_index()
    .drop(['bias^2_exp'],axis=1)                                     # remove redundant bias column
    .rename(columns={'bias^2_real': 'Bias$^2$', 
                     'variance_exp': 'Sampling Variance', 
                     'variance_real': 'Extrapolation Variance'})
)

df['Extrapolation Variance'] = df['Extrapolation Variance'] - df['Sampling Variance']

df = df.melt(                                                       # rows for each stat to allow for stack plot 
    id_vars=['q','nc','beta'], 
    value_vars=['Bias$^2$','Sampling Variance','Extrapolation Variance'],
    var_name='stat', 
    value_name='value')

p = (
    so.Plot(df, x='q', y='value', color='stat')
    .facet(row='beta', col='nc')
    .add(so.Area(), so.Stack())
    .layout(size=(10,8))
)

p.show()
p.save("mse.png")