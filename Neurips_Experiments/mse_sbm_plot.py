import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['toolbar'] = 'None'

data_file = open("SBM/sbm_data.pkl", 'rb')
data = pickle.load(data_file)
data_file.close()

df = pd.DataFrame(data)

sns.set_theme()


colors = ["tab:blue","tab:orange","tab:green"]

f,ax = plt.subplots(1,3, sharex=True, sharey=True)
f.set_figheight(2)
f.set_figwidth(8)
plt.setp(ax, xlim=(min(df['q']),1))
plt.setp(ax, ylim=(0,1))
plt.setp(ax, xlabel='q')
plt.setp(ax[0], ylabel='MSE')

for beta in range(1,4):
    ax[beta-1].set_title(f"$\\beta={beta}$")
    cell_df = df[df['beta'] == beta]
    ax[beta-1].fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], alpha=0.15)
    ax[beta-1].plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)
    ax[beta-1].fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var_s'], color=colors[1], alpha=0.15)
    ax[beta-1].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.5)
    ax[beta-1].fill_between(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], cell_df['bias']**2 + cell_df['var'], color=colors[2], alpha=0.15)
    ax[beta-1].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color='k', linewidth=2)

f.subplots_adjust(bottom=0.25)
plt.show()
f.savefig("mse_sbm_plot.png",bbox_inches='tight')