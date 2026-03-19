from GMMExperiment import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from joblib import delayed, Parallel
import pickle

if __name__ == '__main__':
    generate = True
    save_file = 'figure7.p'
    n_replicates = 50
    T = 80

    n_agents = 30
    n_components = n_agents
    p_mirror = 0
    rag_size = 5

    sigma = 0.2
    epsilon = 1e-12

    n_neighbors = 29

    d_mu_list = np.linspace(1,6,11)

    parameter_dict = {}
    results_dict = {}
    if generate:
        for d_mu in tqdm(d_mu_list):
            # initialization
            initial_weights = epsilon/n_components*np.ones((n_agents, n_components))
            for i in range(n_agents):
                initial_weights[i][i] = 1-epsilon
            gmm_means = np.array([d_mu*i for i in range(n_agents)])
            gmm_stddev = np.array([sigma for _ in range(n_agents)])

            parameter_list = []

            for _ in range(n_replicates):
                initial_gmm_parameters = (initial_weights, gmm_means, 
                                          gmm_stddev)
                parameter_list.append(initial_gmm_parameters)
                f = lambda x: experiment(T, p_mirror, n_neighbors, rag_size, 
                                         initial_weights, gmm_means, 
                                         gmm_stddev)
                parameter_dict[d_mu] = parameter_list
                results_dict[d_mu] = Parallel(n_jobs=-1)(delayed(f)(x) 
                    for x in parameter_dict[d_mu])
        pickle.dump(results_dict, open(save_file, 'wb'))

    df = pd.read_pickle(save_file)

    def threshold(IDs):
        IDs.reverse()
        for index, num in enumerate(IDs):
            if num != 1:
                return T-index
        return -1

    T_threshold = []
    for d in np.linspace(1,6,11)[1:]:
        t_threshold = []
        for v in range(n_replicates):
            gmm_weights_history = df[d][v]
            IDs = []
            for t in range(T+1):
                ids = []
                for i in range(n_agents):
                    ids.append(np.argmax(gmm_weights_history[t][i]))
                IDs.append(ids)
            IDs = pd.DataFrame(IDs)
            silo_count = []
            for t in range(1,IDs.shape[0]):
                silo_count.append(len(set(IDs.iloc[t])))
            t_threshold.append(threshold(silo_count))

        T_threshold.append(t_threshold)

    result = pd.DataFrame(T_threshold)
    Mean = result.mean(axis=1)
    SE = result.std(axis=1)/np.sqrt(n_replicates)
    Top = Mean + 5*SE
    Bottom = Mean - 5*SE
    result['diff_mu'] = np.linspace(1,6,11)[1:]
    result['Average'] = Mean
    result['Top'] = Top
    result['Bottom'] = Bottom

    plt.plot(np.linspace(1,6,11)[1:], Mean, 'k-', linewidth=3)
    plt.fill_between(np.linspace(1,6,11)[1:], Bottom, Top, facecolor='lightgray')
    plt.xlabel('$\Delta \mu$', fontsize=20)
    plt.ylabel('$t^{*}$', fontsize=20)
    plt.savefig('figure7.png')
    plt.show()



        
