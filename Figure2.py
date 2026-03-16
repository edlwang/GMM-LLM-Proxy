from GMMExperiment import *
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # initialization
    p_mirror = 0.4
    T = 400
    n_neighbors = 29
    n_samples_per_interaction = 5 # RAG-size
    sigma = 0.2
    initial_weights = np.zeros((30, 30))
    for i in range(30):
        initial_weights[i][i] = 1
    gmm_means = np.array([i for i in range(30)])
    gmm_stddev = np.array([sigma for _ in range(30)])
    gmm_weights_history = experiment(T, p_mirror, n_neighbors, n_samples_per_interaction, initial_weights, gmm_means,gmm_stddev)
    IDs = []
    for t in range(T+1):
        ids = []
        for i in range(30):
            ids.append(np.argmax(gmm_weights_history[t][i]))
        IDs.append(ids)
    IDsunstable = pd.DataFrame(IDs)
    ax = IDsunstable.plot(y=IDsunstable.columns, title=f'GMM Interaction, p={p_mirror}, k={n_neighbors}, RAG size={n_samples_per_interaction}', color=['darkseagreen']*30, legend=False, xlabel='Time', ylabel='Vertex ID')
    plt.show()

    # Elemetns in silo
    for agent in range(30):
        agent_count = np.sum(IDsunstable.to_numpy() == agent, axis=1)
        plt.plot(range(0,T+1), agent_count, label=f'Agent {agent}')
    plt.title('Unstable Silo')
    plt.xlabel('Time')
    plt.ylabel('Number of Agents in each Silo')
    plt.show()