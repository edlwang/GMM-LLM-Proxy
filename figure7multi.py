from GMMExperiment import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from joblib import delayed, Parallel
import pickle

def run_single_sim(T, p_mirror, n_neighbors, rag_size, weights, means, stddev, seed):
    """
    Runs the experiment and converts the massive history into a single integer
    BEFORE returning it to the main process. 
    """
    # Run the original simulation
    history = multi_experiment(T, p_mirror, n_neighbors, rag_size, weights, means, stddev, seed)
    
    # Convert the 3D array into the convergence time integer
    return time_of_convergence(np.array(history))

if __name__ == '__main__':
    # --- Parameters ---
    generate = True
    save_file = 'figure7_convergence.p'
    n_replicates = 50
    T = 100
    n_agents = 30
    n_components = n_agents
    p_mirror = 0
    rag_size = 5
    sigma = 0.2
    epsilon = 1e-12
    n_neighbors = n_agents - 1
    d_mu_list = np.linspace(1, 6, 11)

    results_dict = {}

    if generate:
        # --- Task Preparation ---
        all_tasks = []
        seed_generator = 42
        for d_mu in d_mu_list:
            # Initialization logic
            initial_weights = epsilon/n_components * np.ones((n_agents, n_components))
            for i in range(n_agents):
                initial_weights[i][i] = 1 - epsilon
            
            gmm_means = (d_mu / np.sqrt(2)) * np.eye(n_agents)
            gmm_stddev = [(sigma**2) * np.eye(n_agents)] * n_agents
            
            for _ in range(n_replicates):
                all_tasks.append((d_mu, initial_weights, gmm_means, gmm_stddev, seed_generator))
                seed_generator += 1

        # --- Parallel Execution ---
        raw_results = Parallel(n_jobs=-1)(
            delayed(run_single_sim)(T, p_mirror, n_neighbors, rag_size, w, m, s, seed) 
            for (d_mu, w, m, s, seed) in tqdm(all_tasks, desc="Running Simulations")
        )

        # --- Reassemble Results ---
        for i, (d_mu, w, m, s, seed) in enumerate(all_tasks):
            if d_mu not in results_dict:
                results_dict[d_mu] = []
            results_dict[d_mu].append(raw_results[i])

        # Save the dictionary of convergence times
        with open(save_file, 'wb') as f:
            pickle.dump(results_dict, f)

    # --- Data Loading & Visualization ---
    data = pd.read_pickle(save_file)
    
    T_threshold = [data[d] for d in d_mu_list[1:]]
    result = pd.DataFrame(T_threshold, index=d_mu_list[1:])

    print("\nConvergence Times (Rows = d_mu, Columns = Replicates):")
    print(result)
    result
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
    plt.savefig('figure7multi.png')
    plt.show()



        
