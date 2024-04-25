import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_file = open("sbm_data_vary_connectivity.pkl", 'rb')
data = pickle.load(data_file)
data_file.close()

df = pd.DataFrame(data)
df['mse'] = df['bias']**2 + df['var']

piis = df['pii'].unique()

sns.set_theme()

colors = ["tab:blue","tab:orange","tab:green"]

f,ax = plt.subplots(1,len(piis), sharex=True, sharey=True)
# plt.setp(ax, xlim=(min(df['q']),1))
# plt.setp(ax, ylim=(0,1))
   
for i,pii in enumerate(df['pii'].unique()):
    cell_df = df[df['pii'] == pii]
    # plt.plot(plot_df['q'],plot_df['mse'],label=pii)

    ax[i].set_title(f"pii={pii}")
    ax[i].fill_between(cell_df['q'], 0, cell_df['bias']**2, color=colors[0], alpha=0.15)
    ax[i].plot(cell_df['q'], cell_df['bias']**2, color=colors[0],alpha=0.5)
    ax[i].fill_between(cell_df['q'], cell_df['bias']**2, cell_df['bias']**2 + cell_df['var_s'], color=colors[1], alpha=0.15)
    ax[i].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], color=colors[1],alpha=0.5)
    ax[i].fill_between(cell_df['q'], cell_df['bias']**2 + cell_df['var_s'], cell_df['bias']**2 + cell_df['var'], color=colors[2], alpha=0.15)
    ax[i].plot(cell_df['q'], cell_df['bias']**2 + cell_df['var'], color=colors[2],alpha=0.5)
#plt.legend()

#sns.lineplot(df,x="q",y="mse",hue="pii",palette="flare")


# limerelplot(
#     data=flights,
#     x="month", y="passengers", col="year", hue="year",
#     kind="line", palette="crest", linewidth=4, zorder=5,
#     col_wrap=3, height=2, aspect=1.5, legend=False,
# )
    
plt.show()
plt.savefig("vary_connectivity_plot.png")
