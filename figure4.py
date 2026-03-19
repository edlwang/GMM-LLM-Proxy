from GMMExperiment import *
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import delayed, Parallel
import pickle

if __name__ == '__main__':
    generate = False
    save_file = 'figure4.p'
    n_replicates = 1
    T = 200

    n_agents_list = [30]

    p_mirror_list = [0.2, 0.5, 0.7]
    rag_size = 5

    sigma = 0.2
    epsilon = 1e-12

    n_neighbors_list_dict = {30: [1,5,15,25]}

    parameter_dict = {}
    results_dict = {}
    if generate:
        for n_agents in tqdm(n_agents_list):
            n_neighbors_list = n_neighbors_list_dict[n_agents]
            n_components = n_agents

            # initialization
            initial_weights = epsilon/n_components*np.ones((n_agents, n_components))
            for i in range(n_agents):
                initial_weights[i][i] = 1-epsilon
            gmm_means = np.array([i for i in range(n_agents)])
            gmm_stddev = np.array([sigma for _ in range(n_agents)])
            """
            gmm_weights_history = experiment(T, p_mirror, n_neighbors, 
                                             rag_size, initial_weights, 
                                             gmm_means, gmm_stddev,
                                             seed=rng.spawn(1))
            """
            parameter_dict[n_agents] = {}
            results_dict[n_agents] = {}
            for p_mirror in tqdm(p_mirror_list):
                parameter_dict[n_agents][p_mirror] = {}
                results_dict[n_agents][p_mirror] = {}
                for n_neighbors in tqdm(n_neighbors_list):
                    interaction_parameters = (p_mirror, n_neighbors, rag_size)

                    parameter_list = []
                    for _ in range(n_replicates):
                        initial_gmm_parameters = (initial_weights, gmm_means, 
                                                  gmm_stddev)
                        parameter_list.append(initial_gmm_parameters)
                    f = lambda x: experiment(T, p_mirror, n_neighbors, rag_size, 
                                             initial_weights, gmm_means, 
                                             gmm_stddev)
                    parameter_dict[n_agents][p_mirror][n_neighbors] = parameter_list
                    results_dict[n_agents][p_mirror][n_neighbors] = Parallel(
                        n_jobs=-1)(delayed(f)(x) 
                        for x in parameter_dict[n_agents][p_mirror][n_neighbors])
        pickle.dump(results_dict, open(save_file, 'wb'))
    results_dict = pickle.load(open('figure4.p', 'rb'))
    rows = [0.2, 0.5, 0.7]
    cols = [1, 5, 15, 25]

    fig, axs = plt.subplots(len(rows), len(cols), figsize=(22,11))
    for pidx, p in enumerate(rows):
        for kidx, k in enumerate(cols):
            gmm_weights_history = results_dict[30][p][k][0]
            IDs = []
            T = 200
            n_agents = 30
            for t in range(T+1):
                ids = []
                for i in range(n_agents):
                    ids.append(np.argmax(gmm_weights_history[t][i]))
                IDs.append(ids)
            IDs = pd.DataFrame(IDs)
            IDs.plot(y=IDs.columns, ax = axs[pidx, kidx], title=f'', color=['darkseagreen']*30, legend=False, xlabel='', ylabel='')
            if pidx != len(rows)-1:
                axs[pidx, kidx].get_xaxis().set_ticks([])
            if kidx != 0:
                axs[pidx, kidx].get_yaxis().set_ticks([])
            axs[pidx, kidx].get_xaxis().set_tick_params(labelsize=15)
            axs[pidx, kidx].get_yaxis().set_tick_params(labelsize=15)
    for ax, col in zip(axs[0], cols):
        ax.set_title(f'k={col}', size='xx-large')
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(f'p={row}', size='xx-large')
    fig.tight_layout()
    plt.savefig('figure4.png')
    plt.show()
