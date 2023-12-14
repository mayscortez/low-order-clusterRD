# Setup
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    graph = "120lat" # standard 120 x 120 lattice graph
    k_fixed = 12
    q_fixed = 1
    B_fixed = 0.1
    beta = 3
    
    title = ['$\\beta={}, N=14400, B={}, q={}$'.format(beta, str(B_fixed), str(q_fixed)),
             '$\\beta={}, N=14400, s={}, B={}$'.format(beta, str(k_fixed), str(B_fixed)),
             '$\\beta={}, N=14400, s={}, B={}$'.format(beta, str(k_fixed), str(B_fixed))]
    fixed =  'k' + str(k_fixed) + '_' + 'B' + str(B_fixed).replace('.','') + '_' + 'q' + str(q_fixed).replace('.','')
    x_label = [fixed+'_'+'clusterSize',
               fixed+ '_' +'clusterFraction', 
               fixed+ '_' + 'newBudget']
    x_var = ['k', 'q', 'p']
    x_plot = ['$s$', '$q$', '$p$']
    model = ['linear']
    for b in model:
        for ind in range(len(x_var)):
            plot(graph,x_var[ind],x_label[ind],b,x_plot[ind],title[ind],beta)


def plot(graph,x_var,x_label,model,x_plot,title,beta=1):
    #load_path = 'outputFiles/' + 'degree' + str(beta) + '/'
    load_path = 'outputFiles/' + 'degree' + str(beta) + '/' + 'newWeights/'
    #save_path = 'plots/' + 'degree' + str(beta) + '/'
    save_path = 'plots/' + 'degree' + str(beta) + '/' + 'newWeights/'
    deg_str = '_deg' + str(beta)
    
    BRD_est = ['PI($p$)', 'LS-Prop', 'LS-Num','DM', 'DM($0.75$)']
    #BRD_est = ['PI($p$)', 'LS-Prop', 'LS-Num']
    #BRD_est = ['PI($p$)']

    experiment = '_'+x_label
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

    sns.lineplot(x=x_var, y='Bias_sq', hue='Estimator', style='Estimator', data=newData, legend='brief', markers=True)
    

    if beta == 1:
        ax.set_ylim(-0.01,0.01)
    elif beta == 2:
        ax.set_ylim(-0.1,0.9)
    else: 
        pass
    
    ax.set_xlabel(x_plot, fontsize = 18)
    ax.set_ylabel("MSE", fontsize = 18)
    ax.set_title(title, fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='lower right', fontsize = 14)
    plt.grid()

    plt.savefig(save_path+graph+experiment+deg_str+'_MSE.pdf')
    plt.close()


if __name__ == "__main__":
    main()
