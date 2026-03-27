from GMMExperiment import *
import pandas as pd
import matplotlib.pyplot as plt
import pickle

if __name__ == '__main__':
    # initialization
    p_mirror = 0.4
    T = 100
    n_agents = 30
    n_components = n_agents
    n_neighbors = 29
    rag_size = 5
    sigma = 0.2
    epsilon = 1e-12
    initial_weights = epsilon/n_components*np.ones((n_agents, n_components))
    for i in range(n_agents):
        initial_weights[i][i] = 1-epsilon
    initial_gmm_means = np.array([[i for i in range(n_agents)]*n_components]).reshape((n_agents, n_components))
    initial_gmm_stddevs = np.array([[sigma for _ in range(n_agents)]*n_components]).reshape((n_agents, n_components))
    gmm_weights_history, gmm_means_history, gmm_stddevs_history = mean_variance_experiment(T, p_mirror, n_neighbors,
                                     rag_size, initial_weights, initial_gmm_means, initial_gmm_stddevs, False, True,
                                     seed=618327487)
    pickle.dump(gmm_weights_history, open('figure2data_variance.p', 'wb'))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    IDs = []
    for t in range(T+1):
        ids = []
        for i in range(n_agents):
            ids.append(np.argmax(gmm_weights_history[t][i]))
        IDs.append(ids)
    IDs = pd.DataFrame(IDs)
    pickle.dump
    IDs.plot(y=IDs.columns, 
                          title=f'GMM Interaction, p={p_mirror}, k={n_neighbors}, RAG size={rag_size}', 
                          color=['darkseagreen']*n_agents, legend=False, 
                          xlabel='Time', ylabel='Vertex ID', ax=ax1)
    """
    # Elements in silo
    for agent in range(n_agents):
        agent_count = np.sum(IDs.to_numpy() == agent, axis=1)
        plt.plot(range(0,T+1), agent_count, label=f'Agent {agent}')
    plt.title('Unstable Silo')
    plt.xlabel('Time')
    plt.ylabel('Number of Agents in each Silo')
    plt.show()
    """
    ans = []
    for t in range(T):
        m = 0
        for i in range(n_agents):
            w = max(gmm_stddevs_history[t][i])
            if w > m:
                m = w
        ans.append(m)
    ax2.plot(ans) 
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Maximum of Standard Deviation among all Agents')
    ax2.set_title('Maximum of Standard Deviation Over Time')

    # Adjust layout so labels don't overlap
    plt.tight_layout()
    plt.savefig("gmm_moving_var_analysis_plots.pdf", format='pdf', bbox_inches='tight')
    plt.show()