from GMMExperiment import *
import pandas as pd
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    # initialization
    p_mirror = 0.4
    T = 100
    n_agents = 2
    n_components = n_agents
    n_neighbors = 1
    rag_size = 5 # RAG-size
    sigma = 0.2
    epsilon = 1e-12
    initial_weights = epsilon/n_components*np.ones((n_agents, n_components))
    for i in range(n_agents):
        initial_weights[i][i] = 1-epsilon
    gmm_means = np.array([[i, 0] for i in range(n_agents)])
    gmm_stddev = np.array([[[sigma,0],[0,sigma]] for _ in range(n_agents)])
    gmm_weights_history = multi_experiment(T, p_mirror, n_neighbors, rag_size, 
                                     initial_weights, gmm_means, gmm_stddev,
                                     seed=15129201)
    pickle.dump(gmm_weights_history, open('figure2multidata.p', 'wb'))
    IDs = []
    for t in range(T+1):
        ids = []
        for i in range(n_agents):
            ids.append(np.argmax(gmm_weights_history[t][i]))
        IDs.append(ids)
    IDs = pd.DataFrame(IDs)
    pickle.dump
    ax = IDs.plot(y=IDs.columns, 
                          title=f'GMM Interaction, p={p_mirror}, k={n_neighbors}, RAG size={rag_size}', 
                          color=['darkseagreen']*n_agents, legend=False, 
                          xlabel='Time', ylabel='Vertex ID')
    plt.show()

    # Elements in silo
    for agent in range(n_agents):
        agent_count = np.sum(IDs.to_numpy() == agent, axis=1)
        plt.plot(range(0,T+1), agent_count, label=f'Agent {agent}')
    plt.title('Unstable Silo')
    plt.xlabel('Time')
    plt.ylabel('Number of Agents in each Silo')
    plt.show()
