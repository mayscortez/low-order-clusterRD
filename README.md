# Covariate Community Correlation
This folder has the files for the experiment varying homophily levels, while keeping other parameters (such as graph structure and treatment probabilities) fixed.
- `main_correlation.py` is the main experiment file
- `execute_main.py` runs the main experiment (this is the file to modify to run your own experiment)
- `experiment_functions.py` contains helper functions for the experiment
- `plots.py` is a script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each model degee)
    - the subfolders `deg1`, `deg2`, `deg3` correspond to different models (with  $\beta=1$, $\beta=2$, and $\beta=3$, respectively.)
    - the subsubfolders `bernoulli` and `complete` denote the design for choosing clusters in the first stage of the two-stage cluster staggered rollout.
- `plots` is a folder containing the resulting plots from `plots.py` (there is a seperate folder for each model degee)
    - folder structure is the same as output

## Running Experiments and Plotting Results
The parameter $\phi$ tunes the balance of clusters with respect to covariates. When $\phi=0$, there is no balance, aka covariate type is perfectly correlated with cluster, and therefore high (perfect) homophily. When $\phi=0.5$, covariates are balanced across clusters, meaning there is low to no homophily. In these experiments, we vary $\phi$ between these two extremes. 

For all experiments, the networks are stochastic block models of size $n=1000$, there are $n_c = 50$ communities/blocks of size $20$, and the expected degree is 10. The parameter $p_{\text{in}}$ tunes the connectivity of the graph and represents the edge probability between units of the same cluster. Given this parameter, the experiment file automatically computes the edge probability between units of different clusters, $p_{\text{out}}$. Given $p_{\text{in}}$, the main experiment file automatically compute $p_{\text{out}}$ such that the expected degree is 10. When $p_{\text{in}}=0.5$, $p_{\text{out}}=0$ and the graph is made of disjoint clusters. When $p_{\text{in}}=0.01$, $p_{\text{out}}=0.01$ and the graph is really Erdos-Renyi.

To run an experiment and save the output, visit the file `execute_main.py`. The file is capable of running several experiments back to back, saving the corresponding output in the corresponding folders. The 6 parameters determining the details of the experiment are listed and explained below:
- `beta`: list of the model degrees
- `B`: list of treatment budgets
- `probs`: list of lists of treatment probabilities
- `designs`: list of strings
- `p_in`: float - the edge probability between units of the same cluster
- `graphNum`: int - the number of graphs to average over in the experiment
- `T`: int - the number of treatment samples to average over per graph

**Example: Run a Single Experiment**

Suppose you want to run a single experiment where the model degree is $\beta=2$, treatment budget is $B=0.06$, treatment probability within chosen clusters is $p=1$, clusters are chosen according to Bernoulli design, we average over 30 SBMs with disjoint clusters and for each graph, average over 30 samples. Then the corresponding parameters are 
- `beta = [2]`
- `B = [0.06]`
- `probs = [[1]]`
- `designs = ["bernoulli"]`
- `p_in = 0.5`
- `graphNum = 30`
- `T = 30`

**Example: Run Multiple Experiments**
Suppose you choose the parameters:
- `beta = [1,2]`
- `B = [0.06, 0.5]`
- `probs = [[0.06, 0.25, 1/3, 2/3, 1], [0.5, 0.625, 25/33, 25/29, 1]]`
- `designs = ["bernoulli"]`
- `p_in = 0.01`
- `graphNum = 30`
- `T = 30`
This runs 5 experiments for a degree $\beta=1$ model and 5 experiments for a degree $\beta-2$ model. Each averages over 30 SBMs, where the edge probability between units of the same cluster and different clusters are the same (aka Erdos-Renyi graphs), and 30 treatment samples per graph. The degree $\beta=1$ experiments all have a treatment budget of $B=0.06$, but each corresponds to a different treatment probability $p \in \{0.06, 0.25, 1/3, 2/3, 1\}$. Note this corresponds to choosing $K=50, 12, 9, 6, 3$ clusters out of $50$, respectively. The degree $\beta=2$ experiments all have a treatment budget of $B=0.5$, but each corresponds to a different treatment probability $p \in \{0.5, 0.625, 25/33, 25/29, 1\}$. Note this corresponds to choosing $K=50, 40, 33, 29, 25$ clusters out of $50$, respectively. 




# Robustness to Misspecification
This folder has the files for the experiments comparing the performance of different estimators and designs under model misspecification.

## Main Robustness to Misspecification experiment files
- `main_vary_p.py` fixes all parameters except the treatment probability within clusters $p$, and computes different, possibly misspecifed, estimators
    - the idea is to compare the robustness of the different designs+estimators as we reduce or increase the extrapolation error
- `execute_main-vary_p.py` is the file to actually run the experiment that varies $p$
- `main_vary_phi.py` fixes all parameters except the balance (w.r.t. covariates) level $\phi$, and computes different, possibly misspecifed, estimators
    - When $\phi=0$, there is high/perfect homophily. When $\phi=0.5$ there is no homophily. The idea is to compare the robustness of the different designs+estimators as we reduce or increase the stregnth of the homophily
- `execute_main-vary_phi.py` is the file to actually run the experiment that varies $\phi$
- `experiment_functions.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network--note this is essentially the same file as the file above, but with some name changes inside the file

## Plotting
`plots_rm_vary_p.py` is a script that plots the output from the experiment `main_vary_p`
- To run this file, scroll down to the bottom. 
- The parameters to change, depending on what you'd like to plot, are `models`, `B`, `Piis`, `Pijs`, `phis`, `cluster_selection`, `type`, and `estimators`
- For example, to plot the MSE as a function of $p$ for the Polynomial Interpolation estimators under cluster staggered rollout and Bernoulli staggered rollout assuming a linear ($\beta=1$) model when the true underlying model is degree 4 polynomial ($\beta=4$), the treatment budget is $B=0.06$, cluster selection is Bernoulli design, clusters are balanced across covariates ($\phi=0.5$), and the graph is made up of disjoint clusters, the parameters would be:
    - `models = [{'type': 'ppom', 'degree':4, 'name': 'ppom4', 'params': []}]`
    - `B = 0.06`
    - `Piis = [0.5]`
    - `Pijs = [0]`
    - `phis = [0.5]`
    - `cluster_selection = "bernoulli"`
    - `type = "MSE"`
    - `estimators = ['PI-$\mathcal{U}(p;1)$','PI-$n(B;1)$']`
- To plot for several different graph structures and balance/homophily levels, note that you can add these all to the lists `Piis`, `Pijs`, `phis`. The code is already set up to iterate over the values in there.
- To see all the possible estimators and models (currently implemented), see the block comment at the bottom of the file `plots_rm_vary_p.py`.

`plots_rm_vary_phi.py` is a script that plots the output from the experiment `main_vary_phi`. 
- The structure is the same as above, except instead of the variable `phis`, you have the variable `probs` corresponding to the treatment probabilities (within clusters) you want to plot for.
- You can plot MSE as a function of $\phi$, Relative Bias as a function of $\phi$, or both. 

## Saving experiment output and plots
- `output` is a folder containing the results from the experiments
    - the subfolders `vary_p` and `vary_phi` contain the results from the respective experiments
    - the subsubfolders `ppom1`, `ppom2`, and `ppom3` correspond to results when the *true* model is polynomial with degree $\beta=1$, $\beta=2$, and $\beta=3$, respectively. If we choose to try other models, such as threshold models, we should create a new folder for those results.
    - the subsubsubfolders `bernoulli` and `complete` denote the design for choosing clusters in the first stage of the two-stage cluster staggered rollout.
- `plots` is a folder containing the resulting plots from `plots.py`
    - folder structure is the same as `output`


# Increasing Edges 
This folder has the files related to the experiment varying the cluster quality with respect to edges crossing clusters.
- `main_increasing_edges.py` is the experiment (fix all parameters, but vary $p_\text{in}$ and $p_\text{out}$, keeping average degree fixed, and compute different estimators)
- `execute_main.py` is the file to actually run the experiment (or several)
- `myFunctions_edges.py` is a file that contains helper/other functions for the experiment e.g. to run the staggered rollout or select clusters or create the network; yes, essentially same file as the corresponding two with the same function in the other folders
- `plots.py` is a script that plots the output from the experiment
- `output` is a folder containing the results from the experiments (there is a seperate folder for each model degee)
- `plots` is a folder containing the resulting plots from `plots.py`(there is a seperate folder for each model degee)


# Misc/Other Notes
- Ignore the folders `Lattice_experiments` and `old-code` in the main file directory.