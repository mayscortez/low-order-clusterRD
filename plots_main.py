# Setup
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

load_path = 'outputFiles/'
save_path = 'plots/'

def main():
    graph = "120lat" # standard 120 x 120 lattice graph
    
    title = ['$\\beta=1, N=14400, B=0.2, q=0.5$','$\\beta=1, N=14400, k=8, B=0.2$','$\\beta=1, N=14400, k=8, B=0.2$']
    x_label = ['clusterSize', 'clusterFraction', 'newBudget']
    x_var = ['k', 'q', 'p']
    x_plot = ['$k$', '$q$', '$p$']
    model = ['linear']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind])
    
    """
    title = ['$\\beta=2, n=5000, k/n=0.5$','$\\beta=2, n=5000, r=1.25$','$\\beta=2, k/n=0.5, r=1.25$']
    x_label = ['ratio', 'tp', 'size']
    x_var = ['ratio', 'p', 'n']
    x_plot = ['$r$', '$k/n$', '$n$']
    model = ['deg2']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind])
    
    title = ['$n=15000, k/n=0.5, r=1.25$']
    x_label = ['varying']
    x_var = ['beta']
    x_plot = ['$\\beta$']
    model = ['deg']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind])
    """


def plot(graph,x_var,x_label,model,x_plot,title,permute=False):
    BRD_est = ['PI($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']

    experiment = '-'+x_label
    print(experiment)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-full-data.csv')
    df = df.assign(Estimator = lambda df: df.Estimator.replace({'PI':'PI($p$)', 'DM(0.75)':'DM($0.75$)'}))

    if experiment == '-varying-deg':
        df = df.loc[df['beta'].isin([0,1,2,3])]

    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)
    newData = df.loc[df['Estimator'].isin(BRD_est)]

    sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=newData, errorbar='sd', legend='brief', markers=True)
    ax.set_ylim(-1,1)
    ax.set_xlabel(x_plot, fontsize = 18)
    ax.set_ylabel("Relative Bias", fontsize = 18)
    ax.set_title(title, fontsize=20)
    handles, labels = ax.get_legend_handles_labels()

    if permute:
        order = [0,3,4,1,2]
        ax.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order], loc='upper right', fontsize = 14)
    else:
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)

    plt.savefig(save_path+graph+experiment+'.pdf')
    plt.close()


if __name__ == "__main__":
    main()
