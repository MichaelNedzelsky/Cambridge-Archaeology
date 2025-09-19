"""
Hypothesis testing framework for inheritance pattern analysis.

Integrates agent-based simulation with ABC model selection to test
5 inheritance systems against archaeological site data.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import euclidean
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json

from data_preprocessing import SiteDataProcessor, load_and_preprocess_data
from inheritance_statistics import InheritancePatternAnalyzer
from agent_simulation import BatchSimulation, SimulationParameters, InheritanceSimulator
from site_parameters import get_site_population, DEFAULT_POPULATION_SIZE
from global_parameters import (
    DISTANCE_WEIGHTS, ABC_ACCEPTANCE_QUANTILE,
    DEFAULT_SIMULATIONS_PER_SYSTEM, INHERITANCE_SYSTEMS
)


class HypothesisTestingFramework:
    """Comprehensive framework for testing inheritance hypotheses."""

    def __init__(self, data_file: str = 'cleaned_dataset.csv',
                 kinship_file: str = 'Cambridshire aDNA summary data.xlsx - DNA kinship details.csv'):
        self.processor = load_and_preprocess_data(data_file, kinship_file)
        self.analyzer = InheritancePatternAnalyzer(self.processor)
        self.inheritance_systems = list(INHERITANCE_SYSTEMS.keys())

    def calculate_summary_statistics(self, simulation_result: Dict) -> Dict:
        """Calculate summary statistics from simulation result."""
        stats = simulation_result['statistics']
        kinship_pairs = simulation_result['kinship_pairs']

        # Basic diversity measures
        summary = {
            'y_diversity': stats.get('y_diversity', 0.0),
            'mt_diversity': stats.get('mt_diversity', 0.0),
            'y_samples': stats.get('y_samples', 0),
            'mt_samples': stats.get('mt_samples', 0),
            'y_unique': stats.get('y_unique', 0),
            'mt_unique': stats.get('mt_unique', 0),
        }

        # Kinship statistics
        if kinship_pairs:
            kinship_df = pd.DataFrame(kinship_pairs)
            total_pairs = len(kinship_pairs)

            # Relationship type proportions
            parent_offspring = len(kinship_df[kinship_df['Relationship'].str.contains('Parent-Offspring', na=False)])
            siblings = len(kinship_df[kinship_df['Relationship'].str.contains('Sibling', na=False)])
            grandparent = len(kinship_df[kinship_df['Relationship'].str.contains('Grandparent', na=False)])

            summary.update({
                'total_kinship_pairs': total_pairs,
                'prop_parent_offspring': parent_offspring / total_pairs if total_pairs > 0 else 0.0,
                'prop_siblings': siblings / total_pairs if total_pairs > 0 else 0.0,
                'prop_grandparent': grandparent / total_pairs if total_pairs > 0 else 0.0,
            })

            # Sex-specific kinship patterns
            father_son = len(kinship_df[
                (kinship_df['Relationship'].str.contains('Parent-Offspring', na=False)) &
                (kinship_df['Sex_1'] == 'M') & (kinship_df['Sex_2'] == 'M')
            ])
            mother_daughter = len(kinship_df[
                (kinship_df['Relationship'].str.contains('Parent-Offspring', na=False)) &
                (kinship_df['Sex_1'] == 'F') & (kinship_df['Sex_2'] == 'F')
            ])

            summary.update({
                'prop_father_son': father_son / total_pairs if total_pairs > 0 else 0.0,
                'prop_mother_daughter': mother_daughter / total_pairs if total_pairs > 0 else 0.0,
            })

            # Haplogroup sharing patterns
            y_matches = len(kinship_df[kinship_df['Y_Match'] == True])
            mt_matches = len(kinship_df[kinship_df['mt_Match'] == True])

            summary.update({
                'prop_y_matches': y_matches / total_pairs if total_pairs > 0 else 0.0,
                'prop_mt_matches': mt_matches / total_pairs if total_pairs > 0 else 0.0,
            })

        else:
            # No kinship pairs found
            summary.update({
                'total_kinship_pairs': 0,
                'prop_parent_offspring': 0.0,
                'prop_siblings': 0.0,
                'prop_grandparent': 0.0,
                'prop_father_son': 0.0,
                'prop_mother_daughter': 0.0,
                'prop_y_matches': 0.0,
                'prop_mt_matches': 0.0,
            })

        # Population structure
        buried_df = simulation_result['buried_individuals']
        if not buried_df.empty:
            total_buried = len(buried_df)
            males = len(buried_df[buried_df['Sex'] == 'M'])
            females = len(buried_df[buried_df['Sex'] == 'F'])
            inheritors = len(buried_df[buried_df['Is_Inheritor'] == True])

            summary.update({
                'sex_ratio': males / females if females > 0 else float('inf'),
                'prop_males': males / total_buried if total_buried > 0 else 0.0,
                'prop_inheritors': inheritors / total_buried if total_buried > 0 else 0.0,
            })

        return summary

    def calculate_observed_statistics(self, site_name: str) -> Dict:
        """Calculate summary statistics from observed site data."""
        signature = self.analyzer.calculate_inheritance_signature(site_name)

        # Convert signature to match simulation output format
        observed_stats = {
            'y_diversity': signature.get('y_diversity', 0.0),
            'mt_diversity': signature.get('mt_diversity', 0.0),
            'y_samples': 0,  # Will be filled from site data
            'mt_samples': 0,
            'y_unique': 0,
            'mt_unique': 0,
            'total_kinship_pairs': signature.get('total_relationships', 0),
            'prop_parent_offspring': 0.0,  # Simplified for now
            'prop_siblings': 0.0,
            'prop_grandparent': 0.0,
            'prop_father_son': signature.get('father_son_ratio', 0.0),
            'prop_mother_daughter': signature.get('mother_daughter_ratio', 0.0),
            'prop_y_matches': 0.0,
            'prop_mt_matches': 0.0,
            'sex_ratio': signature.get('sex_ratio', 1.0),
            'prop_males': signature.get('male_proportion', 0.5),
            'prop_inheritors': 0.5,  # Unknown from observed data
        }

        # Get additional data from site
        site_data = self.processor.get_site_data(site_name)
        if 'y_chr_clean' in site_data.columns:
            y_valid = site_data[site_data['sex_clean'] == 'M']['y_chr_clean'].dropna()
            observed_stats['y_samples'] = len(y_valid)
            observed_stats['y_unique'] = len(y_valid.unique()) if len(y_valid) > 0 else 0

        if 'mt_dna_clean' in site_data.columns:
            mt_valid = site_data['mt_dna_clean'].dropna()
            observed_stats['mt_samples'] = len(mt_valid)
            observed_stats['mt_unique'] = len(mt_valid.unique()) if len(mt_valid) > 0 else 0

        return observed_stats

    def calculate_distance(self, sim_stats: Dict, obs_stats: Dict, weights: Optional[Dict] = None) -> float:
        """Calculate weighted distance between simulated and observed statistics."""
        if weights is None:
            weights = DISTANCE_WEIGHTS.copy()

        distance = 0.0
        total_weight = 0.0

        for key, weight in weights.items():
            if key in sim_stats and key in obs_stats:
                sim_val = sim_stats[key]
                obs_val = obs_stats[key]

                # Handle infinite values
                if np.isinf(sim_val) or np.isinf(obs_val):
                    if np.isinf(sim_val) and np.isinf(obs_val):
                        diff = 0.0
                    else:
                        diff = 10.0  # Large penalty for mismatch
                else:
                    # Normalize by observed value to handle different scales
                    if obs_val != 0:
                        diff = abs(sim_val - obs_val) / abs(obs_val)
                    else:
                        diff = abs(sim_val)

                distance += weight * diff
                total_weight += weight

        return distance / total_weight if total_weight > 0 else float('inf')

    def run_abc_analysis(self, site_name: str, simulation_results: Dict,
                        epsilon: Optional[float] = None, quantile: float = ABC_ACCEPTANCE_QUANTILE) -> Dict:
        """
        Run ABC analysis for a specific site.

        Args:
            site_name: Name of archaeological site
            simulation_results: Dictionary of simulation results by system
            epsilon: Distance threshold (if None, use quantile)
            quantile: Proportion of simulations to accept if epsilon is None
        """
        print(f"Running ABC analysis for {site_name}...")

        # Calculate observed statistics
        observed_stats = self.calculate_observed_statistics(site_name)

        # Calculate distances for all simulations
        distances = []
        systems = []
        sim_stats_list = []

        for system, results in simulation_results.items():
            for result in results:
                if result.get('success', False):
                    sim_stats = self.calculate_summary_statistics(result)
                    distance = self.calculate_distance(sim_stats, observed_stats)

                    distances.append(distance)
                    systems.append(system)
                    sim_stats_list.append(sim_stats)

        distances = np.array(distances)
        systems = np.array(systems)

        if len(distances) == 0:
            raise ValueError("No successful simulations found")

        # Determine acceptance threshold
        if epsilon is None:
            epsilon = np.quantile(distances, quantile)

        # Accept simulations within threshold
        accepted_mask = distances <= epsilon
        accepted_systems = systems[accepted_mask]

        # Calculate posterior probabilities
        unique_systems = np.unique(systems)
        posteriors = {}
        acceptance_counts = {}

        for system in unique_systems:
            system_mask = systems == system
            accepted_system = np.sum(accepted_mask & system_mask)
            total_system = np.sum(system_mask)

            acceptance_counts[system] = {
                'accepted': accepted_system,
                'total': total_system,
                'rate': accepted_system / total_system if total_system > 0 else 0.0
            }

            if len(accepted_systems) > 0:
                posteriors[system] = np.sum(accepted_systems == system) / len(accepted_systems)
            else:
                posteriors[system] = 0.0

        # Calculate Bayes factors
        prior_prob = 1.0 / len(unique_systems)
        bayes_factors = {}
        for system in unique_systems:
            if posteriors[system] > 0:
                bayes_factors[system] = posteriors[system] / prior_prob
            else:
                bayes_factors[system] = 0.0

        # Model comparison metrics
        best_system = max(posteriors, key=posteriors.get)
        best_posterior = posteriors[best_system]

        # Evidence strength interpretation
        if best_posterior > 0.7:
            evidence_strength = "Strong"
        elif best_posterior > 0.5:
            evidence_strength = "Moderate"
        elif best_posterior > 0.3:
            evidence_strength = "Weak"
        else:
            evidence_strength = "Inconclusive"

        results = {
            'site_name': site_name,
            'observed_statistics': observed_stats,
            'posteriors': posteriors,
            'bayes_factors': bayes_factors,
            'acceptance_counts': acceptance_counts,
            'best_system': best_system,
            'best_posterior': best_posterior,
            'evidence_strength': evidence_strength,
            'epsilon': epsilon,
            'n_accepted': np.sum(accepted_mask),
            'n_total': len(distances),
            'distances': distances,
            'systems': systems,
            'accepted_indices': np.where(accepted_mask)[0]
        }

        return results

    def run_batch_simulations(self, site_name: str, n_simulations: int = 100) -> Dict:
        """Run batch simulations for all inheritance systems."""
        print(f"Running {n_simulations} simulations per system for {site_name}...")

        # Map site names to standardized names used in site_parameters
        site_map = {
            'Knobbs 1': 'Knobbs',
            'Knobbs 2': 'Knobbs',
            'Knobbs 3': 'Knobbs',
            'Northwest Cambridge Site IV RB.2C': 'NW_Cambridge',
            'Vicar\'s Farm': 'Vicar_Farm',
            'Fenstanton - Dairy Crest': 'Fenstanton',
            'Fenstanton - Cambridge Road': 'Fenstanton',
            'Duxford': 'Duxford'
        }

        standardized_site = site_map.get(site_name, site_name)

        # Get site-specific population from site_parameters
        try:
            population_size = get_site_population(standardized_site)
            print(f"  Using site-specific population: {population_size} per generation")
        except ValueError:
            # Fall back to default if site not found
            population_size = DEFAULT_POPULATION_SIZE
            print(f"  Site not found in parameters, using default population: {population_size}")

        simulation_results = {}

        for system in self.inheritance_systems:
            print(f"  Simulating {system}...")
            system_results = []

            for i in range(n_simulations):
                params = SimulationParameters(
                    inheritance_system=system,
                    population_per_generation=population_size,
                    site_name=standardized_site  # Pass site name for potential future use
                    # Other parameters use defaults from global_parameters
                )

                try:
                    simulator = InheritanceSimulator(params)
                    result = simulator.run_simulation()
                    system_results.append(result)
                except Exception as e:
                    print(f"    Simulation failed for {system}, run {i}: {e}")
                    continue

            simulation_results[system] = system_results
            print(f"  Completed {len(system_results)} successful simulations for {system}")

        return simulation_results

    def save_results_to_csv(self, site_name: str, abc_results: Dict, simulation_results: Dict):
        """Save analysis results to CSV files."""
        site_clean = site_name.replace(' ', '_').replace('/', '_')

        # 1. Save main ABC results
        main_results = pd.DataFrame([{
            'Site': abc_results['site_name'],
            'Best_System': abc_results['best_system'],
            'Best_Posterior': abc_results['best_posterior'],
            'Evidence_Strength': abc_results['evidence_strength'],
            'N_Accepted': abc_results['n_accepted'],
            'N_Total': abc_results['n_total'],
            'Acceptance_Rate': abc_results['n_accepted'] / abc_results['n_total'],
            'Epsilon': abc_results['epsilon']
        }])

        # Add posterior probabilities for each system
        for system, prob in abc_results['posteriors'].items():
            main_results[f'Posterior_{system}'] = prob

        # Add Bayes factors
        for system, bf in abc_results['bayes_factors'].items():
            main_results[f'BayesFactor_{system}'] = bf

        main_file = f"abc_results_{site_clean}.csv"
        main_results.to_csv(main_file, index=False)

        # 2. Save observed statistics
        obs_stats_df = pd.DataFrame([abc_results['observed_statistics']])
        obs_stats_df['Site'] = site_name
        obs_file = f"observed_stats_{site_clean}.csv"
        obs_stats_df.to_csv(obs_file, index=False)

        # 3. Save simulation summary statistics (aggregated, not raw data)
        sim_summaries = []
        for system, results in simulation_results.items():
            for i, result in enumerate(results):
                if result.get('success', False):
                    summary = self.calculate_summary_statistics(result)
                    summary['system'] = system
                    summary['simulation_id'] = i
                    sim_summaries.append(summary)

        if sim_summaries:
            sim_df = pd.DataFrame(sim_summaries)
            sim_file = f"simulation_summaries_{site_clean}.csv"
            sim_df.to_csv(sim_file, index=False)

            # Also save aggregated stats by system
            agg_stats = sim_df.groupby('system').agg({
                'y_diversity': ['mean', 'std'],
                'mt_diversity': ['mean', 'std'],
                'prop_father_son': ['mean', 'std'],
                'prop_mother_daughter': ['mean', 'std'],
                'sex_ratio': ['mean', 'std']
            }).round(3)
            agg_file = f"system_aggregates_{site_clean}.csv"
            agg_stats.to_csv(agg_file)

        print(f"\nResults saved to CSV files:")
        print(f"  - {main_file} (main ABC results)")
        print(f"  - {obs_file} (observed site statistics)")
        if sim_summaries:
            print(f"  - {sim_file} (simulation summaries)")
            print(f"  - {agg_file} (aggregated statistics by system)")

    def analyze_site(self, site_name: str, n_simulations: int = 100,
                    save_results: bool = True) -> Dict:
        """Complete analysis pipeline for a single site."""
        print(f"\n{'='*60}")
        print(f"ANALYZING SITE: {site_name}")
        print(f"{'='*60}")

        # Run batch simulations
        simulation_results = self.run_batch_simulations(site_name, n_simulations)

        # Run ABC analysis
        abc_results = self.run_abc_analysis(site_name, simulation_results)

        # Print results
        self.print_site_results(abc_results)

        # Save results if requested
        if save_results:
            self.save_results_to_csv(site_name, abc_results, simulation_results)

        return abc_results

    def analyze_all_sites(self, n_simulations: int = 100) -> pd.DataFrame:
        """Analyze all sites and return summary results."""
        sites = self.processor.get_sites()
        all_results = []

        for site in sites:
            try:
                abc_result = self.analyze_site(site, n_simulations, save_results=True)
                all_results.append(abc_result)
            except Exception as e:
                print(f"Failed to analyze site {site}: {e}")
                continue

        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            row = {
                'Site': result['site_name'],
                'Best_System': result['best_system'],
                'Best_Posterior': result['best_posterior'],
                'Evidence_Strength': result['evidence_strength'],
                'N_Accepted': result['n_accepted'],
                'N_Total': result['n_total'],
                'Acceptance_Rate': result['n_accepted'] / result['n_total']
            }

            # Add posterior probabilities
            for system in self.inheritance_systems:
                row[f'Posterior_{system}'] = result['posteriors'].get(system, 0.0)

            # Add Bayes factors
            for system in self.inheritance_systems:
                row[f'BF_{system}'] = result['bayes_factors'].get(system, 0.0)

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('site_analysis_summary.csv', index=False)
        print(f"\nSummary results saved to 'site_analysis_summary.csv'")

        return summary_df

    def print_site_results(self, abc_results: Dict):
        """Print formatted results for a site analysis."""
        print(f"\nRESULTS FOR {abc_results['site_name']}")
        print("-" * 40)
        print(f"Accepted {abc_results['n_accepted']}/{abc_results['n_total']} simulations")
        print(f"Distance threshold (ε): {abc_results['epsilon']:.4f}")

        print(f"\nBest supported system: {abc_results['best_system']}")
        print(f"Posterior probability: {abc_results['best_posterior']:.3f}")
        print(f"Evidence strength: {abc_results['evidence_strength']}")

        print(f"\nPosterior probabilities:")
        for system, prob in abc_results['posteriors'].items():
            print(f"  {system:20s}: {prob:.3f}")

        print(f"\nBayes factors:")
        for system, bf in abc_results['bayes_factors'].items():
            print(f"  {system:20s}: {bf:.2f}")


if __name__ == "__main__":
    # Test the framework with a single site
    framework = HypothesisTestingFramework()

    # Test with Duxford (has good kinship data)
    test_site = "Duxford"
    print(f"Testing framework with {test_site}")

    # Run quick test with fewer simulations
    abc_result = framework.analyze_site(test_site, n_simulations=20)

    print("\nFramework test completed successfully!")