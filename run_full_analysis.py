"""
Run complete analysis of inheritance patterns for all Cambridge Archaeology sites.

This script:
1. Analyzes all sites using the 5 inheritance system hypotheses
2. Runs agent-based simulations for model comparison
3. Uses ABC to determine most likely inheritance pattern for each site
4. Generates comprehensive results and visualizations

Usage:
    python run_full_analysis.py [iterations] [site_name]

Examples:
    python run_full_analysis.py                    # Run all sites with 100 iterations (default)
    python run_full_analysis.py 50                 # Run all sites with 50 iterations
    python run_full_analysis.py 200 Duxford        # Run only Duxford with 200 iterations
    python run_full_analysis.py 100 "Knobbs 1"     # Run only Knobbs 1 with 100 iterations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import argparse
warnings.filterwarnings('ignore')

from hypothesis_testing import HypothesisTestingFramework
from site_parameters import SITE_PARAMETERS
from data_preprocessing import load_and_preprocess_data
from inheritance_statistics import InheritancePatternAnalyzer
from global_parameters import DEFAULT_SIMULATIONS_PER_SYSTEM


def create_results_visualizations(summary_df: pd.DataFrame):
    """Create comprehensive visualizations of the analysis results."""

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cambridge Archaeology Inheritance Pattern Analysis', fontsize=16, fontweight='bold')

    # 1. Best inheritance system by site
    ax = axes[0, 0]
    best_systems = summary_df['Best_System'].value_counts()
    colors = sns.color_palette("husl", len(best_systems))
    best_systems.plot(kind='bar', ax=ax, color=colors)
    ax.set_title('Most Likely Inheritance System by Site Count')
    ax.set_xlabel('Inheritance System')
    ax.set_ylabel('Number of Sites')
    ax.tick_params(axis='x', rotation=45)

    # 2. Evidence strength distribution
    ax = axes[0, 1]
    evidence_counts = summary_df['Evidence_Strength'].value_counts()
    evidence_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
    ax.set_title('Evidence Strength Distribution')
    ax.set_ylabel('')

    # 3. Posterior probability heatmap
    ax = axes[0, 2]
    posterior_cols = [col for col in summary_df.columns if col.startswith('Posterior_')]
    posterior_data = summary_df[['Site'] + posterior_cols].set_index('Site')
    posterior_data.columns = [col.replace('Posterior_', '') for col in posterior_data.columns]

    sns.heatmap(posterior_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
    ax.set_title('Posterior Probabilities by Site')
    ax.set_xlabel('Inheritance System')
    ax.set_ylabel('Site')

    # 4. Acceptance rates
    ax = axes[1, 0]
    summary_df.plot(x='Site', y='Acceptance_Rate', kind='bar', ax=ax)
    ax.set_title('ABC Acceptance Rates by Site')
    ax.set_xlabel('Site')
    ax.set_ylabel('Acceptance Rate')
    ax.tick_params(axis='x', rotation=45)

    # 5. Bayes factors comparison
    ax = axes[1, 1]
    bf_cols = [col for col in summary_df.columns if col.startswith('BF_')]
    bf_data = summary_df[['Site'] + bf_cols].set_index('Site')
    bf_data.columns = [col.replace('BF_', '') for col in bf_data.columns]

    sns.heatmap(bf_data, annot=True, fmt='.1f', cmap='viridis', ax=ax)
    ax.set_title('Bayes Factors by Site')
    ax.set_xlabel('Inheritance System')
    ax.set_ylabel('Site')

    # 6. Best posterior probability by site
    ax = axes[1, 2]
    summary_df.plot(x='Site', y='Best_Posterior', kind='bar', ax=ax, color='skyblue')
    ax.set_title('Best Posterior Probability by Site')
    ax.set_xlabel('Site')
    ax.set_ylabel('Posterior Probability')
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax.legend()

    plt.tight_layout()
    plt.savefig('inheritance_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def generate_detailed_report(summary_df: pd.DataFrame, framework: HypothesisTestingFramework):
    """Generate a detailed text report of the analysis."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CAMBRIDGE ARCHAEOLOGY INHERITANCE PATTERN ANALYSIS - DETAILED REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Executive Summary
    report_lines.append("EXECUTIVE SUMMARY")
    report_lines.append("-" * 40)

    total_sites = len(summary_df)
    best_systems = summary_df['Best_System'].value_counts()
    most_common_system = best_systems.index[0]
    most_common_count = best_systems.iloc[0]

    report_lines.append(f"Total sites analyzed: {total_sites}")
    report_lines.append(f"Most common inheritance pattern: {most_common_system} ({most_common_count} sites)")

    strong_evidence = len(summary_df[summary_df['Evidence_Strength'] == 'Strong'])
    moderate_evidence = len(summary_df[summary_df['Evidence_Strength'] == 'Moderate'])
    weak_evidence = len(summary_df[summary_df['Evidence_Strength'] == 'Weak'])

    report_lines.append(f"Sites with strong evidence: {strong_evidence}")
    report_lines.append(f"Sites with moderate evidence: {moderate_evidence}")
    report_lines.append(f"Sites with weak evidence: {weak_evidence}")
    report_lines.append("")

    # Site-by-site analysis
    report_lines.append("SITE-BY-SITE ANALYSIS")
    report_lines.append("-" * 40)

    for _, row in summary_df.iterrows():
        site_name = row['Site']
        best_system = row['Best_System']
        best_posterior = row['Best_Posterior']
        evidence_strength = row['Evidence_Strength']
        acceptance_rate = row['Acceptance_Rate']

        report_lines.append(f"\n{site_name}:")
        report_lines.append(f"  Best inheritance system: {best_system}")
        report_lines.append(f"  Posterior probability: {best_posterior:.3f}")
        report_lines.append(f"  Evidence strength: {evidence_strength}")
        report_lines.append(f"  ABC acceptance rate: {acceptance_rate:.1%}")

        # Show top 3 systems
        posterior_cols = [col for col in summary_df.columns if col.startswith('Posterior_')]
        site_posteriors = [(col.replace('Posterior_', ''), row[col]) for col in posterior_cols]
        site_posteriors.sort(key=lambda x: x[1], reverse=True)

        report_lines.append(f"  Top alternatives:")
        for i, (system, prob) in enumerate(site_posteriors[:3]):
            if i == 0:
                continue  # Skip the best one as it's already shown
            report_lines.append(f"    {system}: {prob:.3f}")

    # System-specific analysis
    report_lines.append("\n\nSYSTEM-SPECIFIC ANALYSIS")
    report_lines.append("-" * 40)

    for system in ['strongly_patrilineal', 'weakly_patrilineal', 'balanced',
                   'weakly_matrilineal', 'strongly_matrilineal']:
        system_count = len(summary_df[summary_df['Best_System'] == system])
        avg_posterior = summary_df[f'Posterior_{system}'].mean()
        max_posterior = summary_df[f'Posterior_{system}'].max()

        report_lines.append(f"\n{system}:")
        report_lines.append(f"  Best match for {system_count} sites")
        report_lines.append(f"  Average posterior probability: {avg_posterior:.3f}")
        report_lines.append(f"  Maximum posterior probability: {max_posterior:.3f}")

        if system_count > 0:
            best_sites = summary_df[summary_df['Best_System'] == system]['Site'].tolist()
            report_lines.append(f"  Best matching sites: {', '.join(best_sites)}")

    # Statistical summary
    report_lines.append("\n\nSTATISTICAL SUMMARY")
    report_lines.append("-" * 40)

    report_lines.append(f"Mean acceptance rate: {summary_df['Acceptance_Rate'].mean():.1%}")
    report_lines.append(f"Mean best posterior: {summary_df['Best_Posterior'].mean():.3f}")
    report_lines.append(f"Standard deviation of best posterior: {summary_df['Best_Posterior'].std():.3f}")

    high_confidence = len(summary_df[summary_df['Best_Posterior'] > 0.5])
    report_lines.append(f"Sites with >50% posterior for best system: {high_confidence}/{total_sites}")

    # Methodology note
    report_lines.append("\n\nMETHODOLOGY")
    report_lines.append("-" * 40)
    report_lines.append("This analysis used Approximate Bayesian Computation (ABC) to compare")
    report_lines.append("five inheritance systems against archaeological aDNA data:")
    report_lines.append("- Strongly patrilineal: 90% male inheritance")
    report_lines.append("- Weakly patrilineal: 70% male inheritance")
    report_lines.append("- Balanced: 50% male/female inheritance")
    report_lines.append("- Weakly matrilineal: 70% female inheritance")
    report_lines.append("- Strongly matrilineal: 90% female inheritance")
    report_lines.append("")
    report_lines.append("Agent-based simulations generated synthetic cemetery data")
    report_lines.append("under each inheritance system, considering:")
    report_lines.append("- Site-specific population sizes (from archaeological evidence)")
    report_lines.append("- Haplogroup diversity patterns")
    report_lines.append("- Kinship relationship ratios")
    report_lines.append("- Sex-specific burial patterns")
    report_lines.append("- Population genetics constraints")

    # Save report
    report_text = "\n".join(report_lines)
    with open('inheritance_analysis_report.txt', 'w') as f:
        f.write(report_text)

    print("Detailed report saved to 'inheritance_analysis_report.txt'")

    return report_text


def analyze_single_site(framework, site_name, n_simulations):
    """Analyze a single site and display results."""
    print(f"\nAnalyzing {site_name} with {n_simulations} simulations per system...")

    try:
        # Run analysis for single site
        abc_result = framework.analyze_site(site_name, n_simulations, save_results=True)

        # Display results
        print("\n" + "=" * 80)
        print(f"RESULTS FOR {site_name}")
        print("=" * 80)
        print(f"Best inheritance system: {abc_result['best_system']}")
        print(f"Posterior probability: {abc_result['best_posterior']:.3f}")
        print(f"Evidence strength: {abc_result['evidence_strength']}")
        print(f"ABC acceptance rate: {abc_result['n_accepted']}/{abc_result['n_total']} = {abc_result['n_accepted']/abc_result['n_total']:.1%}")

        print("\nPosterior probabilities for all systems:")
        for system, prob in abc_result['posteriors'].items():
            print(f"  {system:25s}: {prob:.3f}")

        return True
    except Exception as e:
        print(f"Error analyzing {site_name}: {e}")
        return False


def map_site_name(user_site, all_sites):
    """
    Map user-provided site name to actual archaeological site name.

    Supports both simplified site names from site_parameters.py and full archaeological names.

    Args:
        user_site (str): Site name provided by user
        all_sites (list): List of all available archaeological site names

    Returns:
        str or list: Mapped site name(s), or None if not found
    """
    if not user_site:
        return None

    # First check if it's already a valid archaeological site name
    if user_site in all_sites:
        return user_site

    # Create mapping from simplified names to archaeological names
    site_mapping = {
        'Duxford': 'Duxford',
        'NW_Cambridge': 'Northwest Cambridge Site IV RB.2C',
        'Vicar_Farm': "Vicar's Farm",
        'Fenstanton': ['Fenstanton - Cambridge Road', 'Fenstanton - Dairy Crest'],
        'Knobbs': ['Knobbs 1', 'Knobbs 2', 'Knobbs 3']
    }

    # Check case-insensitive match with simplified names
    user_site_lower = user_site.lower()
    for simple_name, archaeological_names in site_mapping.items():
        if user_site_lower == simple_name.lower():
            if isinstance(archaeological_names, list):
                # Return all sites for multi-site names like Knobbs and Fenstanton
                return archaeological_names
            else:
                return archaeological_names

    # Check partial matches for multi-site names
    if 'knobbs' in user_site_lower:
        # Check for specific Knobbs site (1, 2, or 3)
        for site in ['Knobbs 1', 'Knobbs 2', 'Knobbs 3']:
            if site.lower() in user_site_lower or user_site_lower in site.lower():
                return site
        # If just "Knobbs", return all Knobbs sites
        return ['Knobbs 1', 'Knobbs 2', 'Knobbs 3']

    if 'fenstanton' in user_site_lower:
        # Check for specific Fenstanton location
        if 'cambridge' in user_site_lower or 'road' in user_site_lower:
            return 'Fenstanton - Cambridge Road'
        elif 'dairy' in user_site_lower or 'crest' in user_site_lower:
            return 'Fenstanton - Dairy Crest'
        # If just "Fenstanton", return both sites
        return ['Fenstanton - Cambridge Road', 'Fenstanton - Dairy Crest']

    # No match found
    return None


def main():
    """Run the complete inheritance pattern analysis."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run inheritance pattern analysis for Cambridge Archaeology sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_analysis.py                    # Run all sites with default iterations
  python run_full_analysis.py 50                 # Run all sites with 50 iterations
  python run_full_analysis.py 200 Duxford        # Run only Duxford with 200 iterations
  python run_full_analysis.py 100 Knobbs         # Run all Knobbs sites (1, 2, 3) with 100 iterations
  python run_full_analysis.py 100 "Knobbs 1"     # Run only Knobbs 1 with 100 iterations
  python run_full_analysis.py 50 Fenstanton      # Run both Fenstanton sites with 50 iterations
  python run_full_analysis.py 100 Vicar_Farm     # Run Vicar's Farm with 100 iterations

Custom inheritance analysis:
  python run_full_analysis.py --female-prob 0.4 100 Duxford    # Custom 40% female inheritance
  python run_full_analysis.py --female_prob 0.8 50 Vicar_Farm  # Custom 80% female inheritance
        """
    )

    parser.add_argument(
        'iterations',
        nargs='?',
        type=int,
        default=DEFAULT_SIMULATIONS_PER_SYSTEM,
        help=f'Number of simulations per inheritance system (default: {DEFAULT_SIMULATIONS_PER_SYSTEM})'
    )

    parser.add_argument(
        'site',
        nargs='?',
        type=str,
        help='Specific site to analyze (optional). If not provided, analyzes all sites.'
    )

    args = parser.parse_args()
    n_simulations = args.iterations
    target_site = args.site

    print("=" * 80)
    print("CAMBRIDGE ARCHAEOLOGY INHERITANCE PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Initialize framework
    print("Initializing analysis framework...")
    framework = HypothesisTestingFramework()

    # Get list of available sites
    all_sites = framework.processor.get_sites()

    if target_site:
        # Map user site name to archaeological site name(s)
        mapped_sites = map_site_name(target_site, all_sites)

        if mapped_sites is None:
            print(f"\nError: Site '{target_site}' not found.")
            print(f"Available sites: {', '.join(all_sites)}")
            print(f"\nSimplified site names you can use:")
            print(f"  - Duxford")
            print(f"  - NW_Cambridge")
            print(f"  - Vicar_Farm")
            print(f"  - Fenstanton (analyzes both Cambridge Road and Dairy Crest)")
            print(f"  - Knobbs (analyzes Knobbs 1, 2, and 3)")
            sys.exit(1)

        # Handle multiple sites (e.g., "Knobbs" maps to all 3 Knobbs sites)
        if isinstance(mapped_sites, list):
            print(f"\nRunning analysis for sites: {', '.join(mapped_sites)}")
            print(f"Simulations per system: {n_simulations}")

            # Analyze each site in the group
            all_success = True
            for site in mapped_sites:
                print(f"\n--- Analyzing {site} ---")
                success = analyze_single_site(framework, site, n_simulations)
                if not success:
                    all_success = False

            if not all_success:
                print("\nSome sites failed to analyze completely.")
                sys.exit(1)
        else:
            # Single site analysis
            print(f"\nRunning analysis for: {mapped_sites}")
            print(f"Simulations per system: {n_simulations}")

            success = analyze_single_site(framework, mapped_sites, n_simulations)

        if success:
            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE!")
            print("=" * 80)
            site_clean = target_site.replace(' ', '_').replace('/', '_')
            print(f"\nResults saved as CSV files:")
            print(f"  - abc_results_{site_clean}.csv (main results)")
            print(f"  - observed_stats_{site_clean}.csv (site statistics)")
            print(f"  - simulation_summaries_{site_clean}.csv (all simulations)")
            print(f"  - system_aggregates_{site_clean}.csv (summary by system)")
    else:
        # Analyze all sites
        print(f"Found {len(all_sites)} sites to analyze:")
        for i, site in enumerate(all_sites, 1):
            print(f"  {i}. {site}")
        print()

        print(f"\nRunning analysis with {n_simulations} simulations per system...")
        print("This may take several minutes...")
        print()

        # Run analysis for all sites
        try:
            summary_df = framework.analyze_all_sites(n_simulations=n_simulations)

            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE!")
            print("=" * 80)

            # Display summary
            print("\nSUMMARY RESULTS:")
            print("-" * 40)
            for _, row in summary_df.iterrows():
                print(f"{row['Site']:30s} -> {row['Best_System']:20s} (p={row['Best_Posterior']:.2f}, {row['Evidence_Strength']})")

            # Generate visualizations
            print("\nGenerating visualizations...")
            create_results_visualizations(summary_df)

            # Generate detailed report
            print("Generating detailed report...")
            generate_detailed_report(summary_df, framework)

            print("\n" + "=" * 80)
            print("ANALYSIS FILES GENERATED:")
            print("=" * 80)
            print("1. site_analysis_summary.csv - Numerical results")
            print("2. inheritance_analysis_visualization.png - Charts and plots")
            print("3. inheritance_analysis_report.txt - Detailed text report")
            print("4. abc_results_[site_name].csv - Individual site ABC results")
            print("5. observed_stats_[site_name].csv - Site statistics")
            print("6. simulation_summaries_[site_name].csv - All simulation data")
            print("7. system_aggregates_[site_name].csv - System-level summaries")
            print("8. inheritance_analysis_results.csv - Statistical signatures")

            print("\nAnalysis completed successfully!")

            return summary_df

        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user.")
            return None
        except Exception as e:
            print(f"\nError during analysis: {e}")
            print("Check the data files and try again.")
            return None


if __name__ == "__main__":
    results = main()