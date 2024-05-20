import pickle
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['toolbar'] = 'None'

data_file = open("Amazon/Experiments/compare_clusterings.pkl", 'rb')
data = pickle.load(data_file)
data_file.close()

df = pd.DataFrame(data)
df = df[df['nc'] == 500]

df['mse'] = df['bias']**2 + df['var']

colors = ["tab:blue","tab:orange","tab:green"]

f,ax = plt.subplots(1,3, sharex=True, sharey=True)
f.set_figheight(2)
f.set_figwidth(8)
plt.setp(ax, xlim=(min(df['q']),1))
plt.setp(ax, ylim=(0,0.3))

ax[1].yaxis.set_tick_params(which='both', labelleft=True)
ax[2].yaxis.set_tick_params(which='both', labelleft=True)

ax[0].set_title("Full Graph Knowledge", fontsize=16)
ax[1].set_title("Covariate Knowledge", fontsize=16)
ax[2].set_title("No Graph Knowledge", fontsize=16)

for i,cl in enumerate(["graph","feature","random"]):
    axis = ax[i]
    cell_df = df[df['clustering'] == cl]

    axis.plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color='k', linewidth=2, label="MSE")
    
    axis.plot(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.5)
    axis.fill_between(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], cell_df['bias']**2 + cell_df['var'], color=colors[2], hatch='\\\\', alpha=0.25,label="Extrapolation Variance")

    axis.plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)
    axis.fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.25,label="Sampling Variance")

    axis.fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], hatch='++', alpha=0.25, label="Bias$^2$")

for axis in ax:
    axis.set_xlabel("q", fontsize=14)
    #axis.set_ylabel("MSE", fontsize=14)

f.subplots_adjust(bottom=0.25)
ax[0].legend(prop={'size': 12})
plt.show()
f.savefig("compare_clusterings_plot.png",bbox_inches='tight')