# Setup
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    graph = "120lat" # standard 120 x 120 lattice graph
    k_fixed = 12
    q_fixed = 0.9
    B_fixed = 0.05
    beta = 2
    
    title = ['$\\beta={}, N=14400, B={}, q={}$'.format(beta, str(B_fixed), str(q_fixed)),
             '$\\beta={}, N=14400, k={}, B={}$'.format(beta, str(k_fixed), str(B_fixed)),
             '$\\beta={}, N=14400, k={}, B={}$'.format(beta, str(k_fixed), str(B_fixed))]
    x_label = [str(q_fixed).replace('.','')+'_'+'clusterSize',
               str(k_fixed)+ '_' +'clusterFraction', 
               str(k_fixed)+ '_' + 'newBudget']
    x_var = ['k', 'q', 'p']
    x_plot = ['$k$', '$q$', '$p$']
    model = ['linear']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind],beta)
    
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


def plot(graph,x_var,x_label,model,x_plot,title,beta=1):
    load_path = 'outputFiles/' + 'degree' + str(beta) + '/'
    save_path = 'plots/' + 'degree' + str(beta) + '/'
    deg_str = '_deg' + str(beta)
    
    #BRD_est = ['PI($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']
    BRD_est = ['PI($p$)']

    experiment = '-'+x_label
    print(experiment)

    # Create and save plots
    df = pd.read_csv(load_path+graph+experiment+'-full-data' + deg_str+ '.csv')
    df = df.assign(Estimator = lambda df: df.Estimator.replace({'PI':'PI($p$)', 'DM(0.75)':'DM($0.75$)'}))

    if experiment == '-varying-deg':
        df = df.loc[df['beta'].isin([0,1,2,3])]

    plt.rc('text', usetex=True)
    
    # Plot with all the estimators
    fig = plt.figure()
    ax = fig.add_subplot(111)
    newData = df.loc[df['Estimator'].isin(BRD_est)]

    #sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', data=df, errorbar='sd', legend='brief', markers=True)
    sns.lineplot(x=x_var, y='Bias', data=df, errorbar='sd', legend='brief', markers=True)
    ax.set_ylim(-1,0.50)
    ax.set_xlabel(x_plot, fontsize = 18)
    ax.set_ylabel("Relative Bias", fontsize = 18)
    ax.set_title(title, fontsize=20)
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)

    plt.savefig(save_path+graph+experiment+deg_str+'.pdf')
    plt.close()


if __name__ == "__main__":
    main()
