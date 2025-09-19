"""
Run complete analysis of inheritance patterns for all Cambridge Archaeology sites.

This script:
1. Analyzes all sites using the 5 inheritance system hypotheses
2. Runs agent-based simulations for model comparison
3. Uses ABC to determine most likely inheritance pattern for each site
4. Generates comprehensive results and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from hypothesis_testing import HypothesisTestingFramework
from data_preprocessing import load_and_preprocess_data
from inheritance_statistics import InheritancePatternAnalyzer
from site_parameters import SITE_PARAMETERS


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


def main():
    """Run the complete inheritance pattern analysis."""

    print("=" * 80)
    print("CAMBRIDGE ARCHAEOLOGY INHERITANCE PATTERN ANALYSIS")
    print("=" * 80)
    print()

    # Initialize framework
    print("Initializing analysis framework...")
    framework = HypothesisTestingFramework()

    # Get list of sites to analyze
    sites = framework.processor.get_sites()
    print(f"Found {len(sites)} sites to analyze:")
    for i, site in enumerate(sites, 1):
        print(f"  {i}. {site}")
    print()

    # Auto-select standard analysis for automated run
    print("Running Standard analysis (100 simulations per system)...")
    n_simulations = 100
    analysis_type = "Standard"

    print(f"\nRunning {analysis_type} analysis with {n_simulations} simulations per system...")
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
        print("4. abc_results_[site_name].pkl - Individual site results")
        print("5. inheritance_analysis_results.csv - Statistical signatures")

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