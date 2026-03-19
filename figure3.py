from GMMExperiment import *
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import delayed, Parallel
import pickle

if __name__ == '__main__':
    rng = np.random.default_rng(1)
    generate = False
    save_file = 'figure3.p'
    n_replicates = 50
    T = 80

    n_agents_list = [30]

    p_mirror_list = [0]
    rag_size = 5

    sigma = 0.2
    epsilon = 1e-12

    n_neighbors_list_dict = {30: [1,2,5,10,15,20,25,27,28,29]}

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
    else:
        df = pd.read_pickle('figure3.p')
        T = 80
        n_agents = 30
        p = 0.0
        k = 10
        # Compute siloes at each time step
        OUTPUT = []
        for p in [0.0]:
            for k in [1,2,5,10,15,20,25,27,28,29]:
                for v in np.arange(50):
                    gmm_weights_history = df[30][p][k][v]
                    IDs = []
                    for t in range(T+1):
                        ids = []
                        for i in range(n_agents):
                            ids.append(np.argmax(gmm_weights_history[t][i]))
                        IDs.append(ids)
                    IDs = pd.DataFrame(IDs)

                    output = [p,k,v,len(set(IDs.iloc[-1])),len(set(IDs.iloc[-2])),len(set(IDs.iloc[-3])),len(set(IDs.iloc[-4]))
                              ,len(set(IDs.iloc[-5])),np.sum(IDs.iloc[-1]==IDs.iloc[-2])/n_agents,np.sum(IDs.iloc[-2]==IDs.iloc[-3])/n_agents
                              ,np.sum(IDs.iloc[-3]==IDs.iloc[-4])/n_agents,np.sum(IDs.iloc[-4]==IDs.iloc[-5])/n_agents]
                    OUTPUT.append(output)
        stat = pd.DataFrame(OUTPUT)
        stat.columns = ['p','k','Replicate','#Silos_T','#Silos_T-1','#Silos_T-2','#Silos_T-3','#Silos_T-4'
                ,'Stability_T','Stability_T-1','Stability_T-2','Stability_T-3']

        # Plot
        df1 = stat[stat['p'] == 0]
        K = df1['k'].values
        Silos = df1['#Silos_T'].values + np.random.normal(0,0.1,len(K))
        mean = pd.DataFrame(np.array(df1.groupby('k').agg({'#Silos_T':'mean'})))[0].values
        se = pd.DataFrame(np.array(df1.groupby('k').agg({'#Silos_T':'std'})/np.sqrt(50)))[0].values
        print(5*se)
        top = mean + 5*se
        bottom = mean - 5*se

        plt.plot([1,2,5,10,15,20,25,27,28,29], mean, 'k-', linewidth=3)
        plt.fill_between([1,2,5,10,15,20,25,27,28,29], bottom, top, facecolor='lightgray')
        plt.scatter(K, Silos, c='mediumslateblue', s=2)
        plt.ylim(0,10.5)
        plt.title('p=0')
        plt.xlabel('k')
        plt.ylabel('Number of Silos')
        plt.savefig('figure3.png')
        plt.show()

        
