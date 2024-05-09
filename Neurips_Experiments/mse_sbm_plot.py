import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['toolbar'] = 'None'

data_file = open("SBM/sbm_data.pkl", 'rb')
data = pickle.load(data_file)
data_file.close()

df = pd.DataFrame(data)

#sns.set_theme()


colors = ["tab:blue","tab:orange","tab:green"]

f,ax = plt.subplots(1,3, sharex=True, sharey=True)
f.set_figheight(2)
f.set_figwidth(8)
plt.setp(ax, xlim=(min(df['q']),1))
plt.setp(ax, ylim=(0,1))

ax[1].yaxis.set_tick_params(which='both', labelleft=True)
ax[2].yaxis.set_tick_params(which='both', labelleft=True)

for beta in range(1,4):
    ax[beta-1].set_title(f"$\\beta={beta}$", fontsize=16)
    cell_df = df[df['beta'] == beta]

    ax[beta-1].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color='k', linewidth=2, label="MSE")
    
    ax[beta-1].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.5)
    ax[beta-1].fill_between(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], cell_df['bias']**2 + cell_df['var'], color=colors[2], hatch='\\\\', alpha=0.25,label="Extrapolation Variance")

    ax[beta-1].plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)
    ax[beta-1].fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.25,label="Sampling Variance")

    ax[beta-1].fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], hatch='++', alpha=0.25, label="Bias$^2$")

for axis in ax:
    axis.set_xlabel("q", fontsize=14)
    #axis.set_ylabel("MSE", fontsize=14)

f.subplots_adjust(bottom=0.25)
ax[0].legend(prop={'size': 12})
plt.show()
f.savefig("mse_sbm_plot.png",bbox_inches='tight')