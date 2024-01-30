# plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
plt.rcParams.update({
    "text.usetex": True#,
    #"font.family": "serif",
    #"font.sans-serif": "Helvetica",
})
plt.rcParams["mathtext.fontset"]

def main(model, p=0.06, p_in=0.5, phi=0, cluster_selection = "bernoulli", plot_type='MSE', estimators=['PI$(q;1)$', 'E[PI$(q;1)|\mathcal{U}$]']): 
    degree = model['degree']
    model_name = model['name']
    experiment = 'vary_p'
    if model["type"] == 'ppom':
        my_path = experiment + '/'  + model_name + '/' + cluster_selection + '/' 
    else:            
        my_path = experiment + '/'  + model_name + '-ppom' + str(degree) + '/' + cluster_selection + '/'           

    n = 1000
    nc = 50
    p_out = (0.5-p_in)/49
    
    if model["type"] == 'threshold' and model["params"][1] == "prop":
        name = '$\mathrm{Prop}_i(\mathbf{z};' + str(model["params"][0]) +  ')$'
    elif model["type"] == 'threshold' and model["params"][1] == "num":
        name = '$\mathrm{Num}_i(\mathbf{z};' + str(model["params"][0]) +  ')$'
    elif model["type"] == 'saturation':
        name = '$\mathrm{Sat}_i(\mathbf{z};' + str(model["params"][0]) +  ')$'
    elif model["type"] == 'ppom':
        name = 'Low-Order Polynomial'
    else:
        raise ValueError("Model type is invalid.")


    fixed = '_n' + str(n) + '_nc' + str(nc) + '_' + 'in' + str(np.round(p_in,3)).replace('.','') + '_out' + str(np.round(p_out,3)).replace('.','') + '_p' + str(p).replace('.','') + '_phi' + str(phi).replace('.','') # naming convention
    
    x_label = [experiment + '-' + model_name + fixed + '_' +  cluster_selection]
    x_var = ['q']
    x_plot = ['treatment probability $q$']
    title = ['True Model: {} with $\\beta={}$'.format(name, degree)] # \n ER$({},{}), p={}, \phi={}$
    print(title[0])
    for ind in range(len(x_var)):
        plot(my_path, x_var[ind],x_label[ind],x_plot[ind],title[ind],estimators, plot_type)

def plot(my_path, x_var, experiment_label, x_plot, title, estimators, plot_type='both'):
    save_path = 'plots/' + my_path

    color_map = {'PI-$n(p;1)$': '#6a9f00', 'PI-$n(p;2)$':'#b2ce02', 'PI-$n(p;3)$': '#69cc00',
                 'PI-$\mathcal{U}(p;1)$': '#1b45a6', 'PI-$\mathcal{U}(p;2)$': '#019cca', 'PI-$\mathcal{U}(p;3)$': '#009d9d', 
                 'HT': '#e51e31',
                 'DM-C': '#e35610', 'DM-C($0.75$)': '#ff7787', #'DM-C': '#fb5082'
                 'PI-$n(B;1)$': '#e0d100', 'PI-$n(B;2)$': '#ffa706', 'PI-$n(B;3)$': '#ed5902',
                 'LS-Prop(1)': '#009633', 'LS-Prop(2)': '#95c413', 'LS-Prop(3)': '#f5c003',
                 'LS-Num(1)': '#46c1c1', 'LS-Num(2)': '#3e66c9', 'LS-Num(3)': '#c42796', 
                 'DM': '#9e35af', 'DM($0.75$)': '#c069c9'}
    
    og_color_map = {'PI$(q;1)$': '#0083ff', 'PI$(q;2)$':'#ff0083', 'PI$(q;3)$': '#83ff00',
                 'PI-$\mathcal{U}(q;1)$': '#0d7901', 'PI-$\mathcal{U}(q;2)$': '#5bd94e', 'PI-$\mathcal{U}(q;3)$': '#085001', 
                 'HT': '#1100ff',
                 'DM-C': '#990002', 'DM-C($0.75$)': '#5f0099',
                 'PI$(p;1)$': '#0296fb', 'PI$(p;2)$': '#4eb6fc', 'PI$(p;3)$': '#0169b0',
                 'LS-Prop(1)': '#fb6702', 'LS-Prop(2)': '#fb6702', 'LS-Prop(3)': '#fb6702',
                 'LS-Num(1)': '#15c902', 'LS-Num(2)': '#15c902', 'LS-Num(3)': '#15c902', 
                 'DM': '#ff0004', 'DM($0.75$)': '#9e00ff'}
    
    original_color_map = {'PI($p$)': '#1e77b4',
                'PI($k/n$)': '#1e81b0',
                'PI($\hat{k}/n$)': '#38b01e',
                'LS-Prop': '#fe8e29',
                'LS-Num': '#2ca02b',
                'DM': '#d62727',
                'DM($0.75$)':'#9368bd'}
    
    color_pal = [og_color_map[est] for est in estimators]

    print('\n'+experiment_label+'_'+'vary-'+x_var)

    #Create and save plots
    df = pd.read_csv('output/' + my_path + experiment_label + '-full.csv')
    newData = df.loc[df['Estimator'].isin(estimators)]

    plt.rc('text', usetex=True)
    
    if (plot_type == 'MSE' or plot_type == 'both'):
        # MSE plots
        fig = plt.figure()
        ax = fig.add_subplot(111)

        sns.lineplot(x=x_var, y='Rel_bias_sq', hue='Estimator', style='Estimator', errorbar=None, data=newData, legend='brief', markers=True, palette=color_pal)

        #ax.set_xlim(0,0.001)
        ax.set_ylim(0,1.05)
        ax.set_xlabel(x_plot, fontsize = 18)
        ax.set_ylabel("MSE", fontsize = 18)
        ax.set_title(title, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper left', fontsize = 14)
        #plt.grid()
        plt.tight_layout()

        save_str = save_path + experiment_label + '_MSE.png'
        plt.savefig(save_str)
        print("saved: " + save_str)
        plt.close()

    if (plot_type == 'Bias' or plot_type == 'both'):
        # Plot with all the estimators
        fig = plt.figure()
        ax = fig.add_subplot(111)

        sns.lineplot(x=x_var, y='Bias', hue='Estimator', style='Estimator', errorbar='sd', data=newData, legend='brief', markers=True, palette=color_pal)

        #ax.set_xlim(0,0.001)
        ax.set_ylim(-2,2)
        ax.set_xlabel(x_plot, fontsize = 18)
        ax.set_ylabel("Relative Bias", fontsize = 18)
        ax.set_title(title, fontsize=18)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper right', fontsize = 14)
        #plt.grid()
        plt.tight_layout()

        save_str = save_path + experiment_label + '.png'
        plt.savefig(save_str)
        print("saved: " + save_str)
        plt.close()
    

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    theta_prop = 0.5
    theta_num = 5
    theta = 12 # for deg3, 34; for deg 4, 42
    models = [{'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []}]
    budget = 0.06        
    Piis = [0.4]    # edge probability within clusters
    phis = [0,0.5] # covariate balance parameter (phi = 0 is exact homophily, phi = 0.5 is no homophily)
    cluster_selection = "bernoulli" # other option: "complete" (for how to choose clusters)
    plot_type = "MSE"   # other options:  "Bias"   "MSE" (what type of plot do you want to make)
    estimators = ['PI$(q;1)$', 'PI$(q;2)$', 'PI$(q;3)$']   # which estimators to plot

    for i in range(len(models)):
        print('Plotting for true model: {} ({} design)'.format(models[i]['name'],cluster_selection))
        for j in range(len(Piis)):
            for phi in phis:
                main(models[i], budget, Piis[j], phi, cluster_selection, plot_type, estimators)
        print() 

'''
theta_prop = 0.5
theta_num = 5
theta = 12 # for deg3, 34; for deg 4, 42
models = [{'type': 'ppom', 'degree':1, 'name':'ppom1', 'params': []},
            {'type': 'ppom', 'degree':2, 'name': 'ppom2', 'params': []},
            {'type': 'ppom', 'degree':3, 'name': 'ppom3', 'params': []},
            {'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []},'
            {'type': 'threshold', 'degree': 2, 'name': 'threshold_prop_' + str(theta_prop).replace(".", ""), 'params': [theta_prop, 'prop']},
            {'type': 'threshold', 'degree': 2, 'name': 'threshold_num_' + str(theta_num), 'params': [theta_num, 'num']},
            {'type': 'saturation', 'degree': 2, 'name': 'saturation_' + str(theta), 'params': [theta]}]
Piis = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.01]
phis = [0, 0.1, 0.2, 0.3, 0.4, 0.5] 

Possible estimators: 
    names_ClRD = ['PI-$n(p;1)$', 'PI-$\mathcal{U}(p;1)$', 'PI-$n(p;2)$', 'PI-$\mathcal{U}(p;2)$', 'PI-$n(p;3)$', 'PI-$\mathcal{U}(p;3)$', 'HT', 'DM-C', 'DM-C($0.75$)']
    names_BRD = ['PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)', 'PI-$n(B;2)$', 'LS-Prop(2)', 'LS-Num(2)', 'PI-$n(B;3)$', 'LS-Prop(3)', 'LS-Num(3)','DM', 'DM($0.75$)']
NOTE: for the colors to be consistent, the estimators have to be in a specific order, regardless of which are included
NOTE (cont): they are in order in names_ClRD and names_BRD, so as long as that relative order is kept and cluster design-based ones come before bernoulli design-based ones, it should be fine

Examples:
    estimators = ['PI-$\mathcal{U}(p;1)$', 'PI-$\mathcal{U}(p;2)$', 'PI-$\mathcal{U}(p;3)$', 'HT', 'PI-$n(B;1)$', 'LS-Prop(1)', 'DM($0.75$)'] 
    estimators = ['PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)']
    estimators = ['PI-$n(p;1)$', 'PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)'] 
    estimators = ['PI-$\mathcal{U}(p;1)$', 'HT', 'DM-C', 'DM-C($0.75$)', 'PI-$n(B;1)$', 'LS-Prop(1)', 'LS-Num(1)']  
    estimators = ['PI-$\mathcal{U}(p;1)$', 'PI-$\mathcal{U}(p;2)$', 'PI-$\mathcal{U}(p;3)$', 'PI-$n(B;1)$', 'PI-$n(B;2)$'] 
    estimators = ['PI-$\mathcal{U}(p;1)$', 'PI-$\mathcal{U}(p;2)$', 'PI-$\mathcal{U}(p;3)$', 'HT', 'DM-C', 'DM-C($0.75$)']  
'''