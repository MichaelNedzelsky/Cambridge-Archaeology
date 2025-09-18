"""
Approximate Bayesian Computation (ABC) for model selection.

Compares simulated data against observed Cambridge archaeological data
to determine which social organization scenario is most probable.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.spatial.distance import euclidean
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from calculate_statistics import calculate_observed_statistics


def normalize_statistics(stats_dict):
    """
    Convert statistics dictionary to normalized vector for distance calculation.
    """
    vector = np.array([
        stats_dict['mean_y_diversity'],
        stats_dict['mean_mt_diversity'],
        stats_dict['fst_y'],
        stats_dict['fst_mt'],
        stats_dict['prop_father_son'],
        stats_dict['prop_mother_daughter'],
        stats_dict['prop_siblings']
    ])
    return vector


def calculate_distance(sim_stats, obs_stats, weights=None):
    """
    Calculate weighted Euclidean distance between simulated and observed statistics.

    weights: Optional array of weights for each statistic
    """
    sim_vector = normalize_statistics(sim_stats)
    obs_vector = normalize_statistics(obs_stats)

    if weights is not None:
        sim_vector *= weights
        obs_vector *= weights

    return euclidean(sim_vector, obs_vector)


def abc_rejection(simulation_results, observed_stats, epsilon=None, quantile=0.1):
    """
    ABC rejection algorithm for model selection.

    simulation_results: Dictionary of results by scenario
    observed_stats: Observed statistics dictionary
    epsilon: Distance threshold (if None, use quantile)
    quantile: If epsilon is None, accept this proportion of simulations
    """
    # Calculate distances for all simulations
    distances = []
    scenarios = []

    for scenario, results in simulation_results.items():
        for result in results:
            if result['success']:
                dist = calculate_distance(
                    result['statistics']['summary_vector'],
                    observed_stats['summary_vector']
                )
                distances.append(dist)
                scenarios.append(scenario)

    distances = np.array(distances)
    scenarios = np.array(scenarios)

    # Determine acceptance threshold
    if epsilon is None:
        epsilon = np.quantile(distances, quantile)

    # Accept simulations within threshold
    accepted = distances <= epsilon
    accepted_scenarios = scenarios[accepted]

    # Calculate posterior probabilities
    unique_scenarios = np.unique(scenarios)
    posteriors = {}

    for scenario in unique_scenarios:
        if len(accepted_scenarios) > 0:
            posteriors[scenario] = np.sum(accepted_scenarios == scenario) / len(accepted_scenarios)
        else:
            posteriors[scenario] = 0.0

    # Calculate Bayes factors relative to uniform prior
    prior_prob = 1.0 / len(unique_scenarios)
    bayes_factors = {}
    for scenario in unique_scenarios:
        if posteriors[scenario] > 0:
            bayes_factors[scenario] = posteriors[scenario] / prior_prob
        else:
            bayes_factors[scenario] = 0.0

    return {
        'posteriors': posteriors,
        'bayes_factors': bayes_factors,
        'epsilon': epsilon,
        'n_accepted': np.sum(accepted),
        'n_total': len(distances),
        'distances': distances,
        'scenarios': scenarios,
        'accepted_indices': np.where(accepted)[0]
    }


def plot_abc_results(abc_results, output_file='abc_results.png'):
    """
    Create visualizations of ABC results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Posterior probabilities
    ax = axes[0, 0]
    scenarios = list(abc_results['posteriors'].keys())
    probs = [abc_results['posteriors'][s] for s in scenarios]
    ax.bar(scenarios, probs)
    ax.set_ylabel('Posterior Probability')
    ax.set_title('Model Selection: Posterior Probabilities')
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for i, (s, p) in enumerate(zip(scenarios, probs)):
        ax.text(i, p + 0.02, f'{p:.2f}', ha='center')

    # 2. Distance distributions by scenario
    ax = axes[0, 1]
    for scenario in scenarios:
        scenario_dists = abc_results['distances'][abc_results['scenarios'] == scenario]
        ax.hist(scenario_dists, alpha=0.5, label=scenario, bins=30)
    ax.axvline(abc_results['epsilon'], color='red', linestyle='--',
              label=f'ε = {abc_results["epsilon"]:.3f}')
    ax.set_xlabel('Distance from Observed Data')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distributions')
    ax.legend()

    # 3. Bayes factors
    ax = axes[1, 0]
    bf_values = [abc_results['bayes_factors'][s] for s in scenarios]
    ax.bar(scenarios, bf_values)
    ax.set_ylabel('Bayes Factor')
    ax.set_title('Bayes Factors (relative to uniform prior)')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

    # 4. Acceptance rates
    ax = axes[1, 1]
    acceptance_rates = []
    for scenario in scenarios:
        scenario_mask = abc_results['scenarios'] == scenario
        n_scenario = np.sum(scenario_mask)
        n_accepted = np.sum(scenario_mask & (abc_results['distances'] <= abc_results['epsilon']))
        rate = n_accepted / n_scenario if n_scenario > 0 else 0
        acceptance_rates.append(rate)

    ax.bar(scenarios, acceptance_rates)
    ax.set_ylabel('Acceptance Rate')
    ax.set_title('Proportion of Simulations Accepted')
    ax.set_ylim([0, max(acceptance_rates) * 1.2])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

    return fig


def compare_statistics(simulation_results, observed_stats):
    """
    Create detailed comparison of statistics between scenarios and observed data.
    """
    # Collect statistics by scenario
    scenario_stats = defaultdict(lambda: defaultdict(list))

    for scenario, results in simulation_results.items():
        for result in results:
            if result['success']:
                for key, value in result['statistics']['summary_vector'].items():
                    scenario_stats[scenario][key].append(value)

    # Create comparison DataFrame
    comparison = []
    stat_names = list(observed_stats['summary_vector'].keys())

    for stat_name in stat_names:
        row = {'statistic': stat_name, 'observed': observed_stats['summary_vector'][stat_name]}

        for scenario in scenario_stats.keys():
            values = scenario_stats[scenario][stat_name]
            row[f'{scenario}_mean'] = np.mean(values) if values else np.nan
            row[f'{scenario}_std'] = np.std(values) if values else np.nan

        comparison.append(row)

    comparison_df = pd.DataFrame(comparison)

    # Calculate which scenario is closest for each statistic
    for _, row in comparison_df.iterrows():
        obs_val = row['observed']
        distances = {}
        for scenario in scenario_stats.keys():
            mean_val = row[f'{scenario}_mean']
            if not pd.isna(mean_val):
                distances[scenario] = abs(mean_val - obs_val)

        if distances:
            closest = min(distances, key=distances.get)
            row['closest_scenario'] = closest

    return comparison_df


def run_abc_analysis(batch_file, observed_data_file='combined_grouped.csv',
                     epsilon=None, quantile=0.05):
    """
    Run complete ABC analysis on simulation results.

    batch_file: Path to pickle file with simulation results
    observed_data_file: Path to observed Cambridge data CSV
    epsilon: Distance threshold (if None, use quantile)
    quantile: Proportion of simulations to accept
    """
    # Load simulation results
    print("Loading simulation results...")
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f)

    simulation_results = batch_data['results']

    # Calculate observed statistics
    print("Calculating observed statistics...")
    observed_stats = calculate_observed_statistics(observed_data_file)

    # Run ABC
    print("Running ABC rejection algorithm...")
    abc_results = abc_rejection(simulation_results, observed_stats,
                               epsilon=epsilon, quantile=quantile)

    # Print results
    print("\n" + "="*50)
    print("ABC ANALYSIS RESULTS")
    print("="*50)

    print(f"\nAccepted {abc_results['n_accepted']}/{abc_results['n_total']} simulations")
    print(f"Distance threshold (ε): {abc_results['epsilon']:.4f}")

    print("\nPOSTERIOR PROBABILITIES:")
    for scenario, prob in abc_results['posteriors'].items():
        print(f"  {scenario:15s}: {prob:.3f}")

    print("\nBAYES FACTORS (vs uniform prior):")
    for scenario, bf in abc_results['bayes_factors'].items():
        print(f"  {scenario:15s}: {bf:.2f}")

    # Interpret results
    best_scenario = max(abc_results['posteriors'], key=abc_results['posteriors'].get)
    best_prob = abc_results['posteriors'][best_scenario]

    print(f"\nBEST SUPPORTED SCENARIO: {best_scenario} (probability = {best_prob:.3f})")

    if best_prob > 0.7:
        print("Strong evidence for this scenario")
    elif best_prob > 0.5:
        print("Moderate evidence for this scenario")
    else:
        print("Weak evidence - multiple scenarios are plausible")

    # Create comparison table
    print("\n" + "="*50)
    print("STATISTICAL COMPARISON")
    print("="*50)
    comparison_df = compare_statistics(simulation_results, observed_stats)
    print("\n", comparison_df.to_string(index=False))

    # Create visualizations
    # print("\nCreating visualizations...")
    # plot_abc_results(abc_results)  # Skip plotting for now

    return abc_results, comparison_df


if __name__ == "__main__":
    # Look for most recent batch file
    results_dir = Path("simulation_results")
    if results_dir.exists():
        batch_files = list(results_dir.glob("batch_*.pkl"))
        if batch_files:
            latest_batch = max(batch_files, key=lambda p: p.stat().st_mtime)
            print(f"Using batch file: {latest_batch}")

            results = run_abc_analysis(
                latest_batch,
                observed_data_file='combined_grouped.csv',
                quantile=0.05  # Accept top 5% of simulations
            )
        else:
            print("No simulation results found. Run run_experiments.py first.")
    else:
        print("No simulation_results directory found. Run run_experiments.py first.")