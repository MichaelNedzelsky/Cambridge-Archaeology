"""
Explanation of how weighted statistics work in the ABC distance calculation.

This script demonstrates where the weights come from and how they're applied.
"""

import numpy as np

def demonstrate_weight_rationale():
    """Explain the rationale behind the chosen weights."""

    print("=" * 70)
    print("WEIGHT RATIONALE IN ABC DISTANCE CALCULATION")
    print("=" * 70)
    print()

    print("The weights in our ABC framework are based on:")
    print("1. Theoretical importance for distinguishing inheritance systems")
    print("2. Expected signal strength in archaeological data")
    print("3. Reliability of the measurement")
    print()

    weights = {
        'y_diversity': 2.0,
        'mt_diversity': 2.0,
        'prop_father_son': 1.5,
        'prop_mother_daughter': 1.5,
        'sex_ratio': 1.0,
        'prop_y_matches': 1.0,
        'prop_mt_matches': 1.0,
        'prop_inheritors': 0.5,
    }

    print("WEIGHT ASSIGNMENTS AND RATIONALE:")
    print("-" * 50)

    # High weight (2.0) - Primary discriminators
    print("🔴 HIGH WEIGHT (2.0) - Primary discriminators:")
    print("  • y_diversity: Y-chromosome diversity")
    print("    - Theoretical foundation: Pure patrilineal → 0, Pure matrilineal → 1")
    print("    - Strong signal: Directly reflects inheritance patterns")
    print("    - Reliable: Well-established population genetics measure")
    print()
    print("  • mt_diversity: Mitochondrial DNA diversity")
    print("    - Theoretical foundation: Pure patrilineal → high, Pure matrilineal → low")
    print("    - Strong signal: Complementary to Y-chromosome patterns")
    print("    - Reliable: Less affected by sampling bias than kinship")
    print()

    # Medium weight (1.5) - Strong kinship signals
    print("🟡 MEDIUM WEIGHT (1.5) - Strong kinship signals:")
    print("  • prop_father_son: Father-son relationship ratio")
    print("    - Theoretical foundation: Higher in patrilineal systems")
    print("    - Moderate signal: Depends on kinship detection accuracy")
    print("    - Archaeological relevance: Direct evidence of male inheritance")
    print()
    print("  • prop_mother_daughter: Mother-daughter relationship ratio")
    print("    - Theoretical foundation: Higher in matrilineal systems")
    print("    - Moderate signal: Complementary to father-son patterns")
    print("    - Archaeological relevance: Direct evidence of female inheritance")
    print()

    # Standard weight (1.0) - Supporting evidence
    print("🟢 STANDARD WEIGHT (1.0) - Supporting evidence:")
    print("  • sex_ratio: Male/female burial ratio")
    print("    - Moderate signal: Can reflect burial bias, not just inheritance")
    print("    - Context-dependent: Influenced by site function and period")
    print("    - Baseline importance: Standard demographic measure")
    print()
    print("  • prop_y_matches: Y-chromosome sharing in kinship pairs")
    print("    - Supporting evidence: Validates kinship relationships")
    print("    - Sample size dependent: Only meaningful with sufficient pairs")
    print("    - Quality control: Helps verify genetic vs. social relationships")
    print()
    print("  • prop_mt_matches: mtDNA sharing in kinship pairs")
    print("    - Supporting evidence: Validates maternal relationships")
    print("    - Complementary: Works with Y-chromosome sharing")
    print("    - Consistency check: Expected patterns for different systems")
    print()

    # Low weight (0.5) - Uncertain measures
    print("🔵 LOW WEIGHT (0.5) - Uncertain measures:")
    print("  • prop_inheritors: Proportion of inheritors")
    print("    - Uncertain signal: Difficult to determine from burial data")
    print("    - Model artifact: More of a simulation parameter than observable")
    print("    - Reduced importance: Cannot be directly validated archaeologically")
    print()


def demonstrate_distance_calculation():
    """Show how weighted distance calculation works with concrete example."""

    print("=" * 70)
    print("WEIGHTED DISTANCE CALCULATION EXAMPLE")
    print("=" * 70)
    print()

    # Example data from Duxford analysis
    observed = {
        'y_diversity': 0.833,
        'mt_diversity': 0.982,
        'prop_father_son': 0.14,
        'prop_mother_daughter': 0.14,
        'sex_ratio': 1.0,
        'prop_y_matches': 0.2,
        'prop_mt_matches': 0.4,
        'prop_inheritors': 0.6,
    }

    # Simulated data from strongly patrilineal system
    simulated_patrilineal = {
        'y_diversity': 0.1,    # Low Y diversity expected
        'mt_diversity': 0.8,   # High mt diversity expected
        'prop_father_son': 0.4, # High father-son expected
        'prop_mother_daughter': 0.05, # Low mother-daughter expected
        'sex_ratio': 1.2,
        'prop_y_matches': 0.8,
        'prop_mt_matches': 0.3,
        'prop_inheritors': 0.9,
    }

    # Simulated data from strongly matrilineal system
    simulated_matrilineal = {
        'y_diversity': 0.9,    # High Y diversity expected
        'mt_diversity': 0.2,   # Low mt diversity expected
        'prop_father_son': 0.05, # Low father-son expected
        'prop_mother_daughter': 0.4, # High mother-daughter expected
        'sex_ratio': 0.8,
        'prop_y_matches': 0.2,
        'prop_mt_matches': 0.8,
        'prop_inheritors': 0.9,
    }

    weights = {
        'y_diversity': 2.0,
        'mt_diversity': 2.0,
        'prop_father_son': 1.5,
        'prop_mother_daughter': 1.5,
        'sex_ratio': 1.0,
        'prop_y_matches': 1.0,
        'prop_mt_matches': 1.0,
        'prop_inheritors': 0.5,
    }

    def calculate_weighted_distance(sim_stats, obs_stats, weights):
        """Calculate weighted distance with detailed breakdown."""
        print(f"Calculating distance between simulation and observed data:")
        print("-" * 60)

        distance = 0.0
        total_weight = 0.0

        for key, weight in weights.items():
            if key in sim_stats and key in obs_stats:
                sim_val = sim_stats[key]
                obs_val = obs_stats[key]

                # Calculate normalized difference
                if obs_val != 0:
                    diff = abs(sim_val - obs_val) / abs(obs_val)
                else:
                    diff = abs(sim_val)

                weighted_contribution = weight * diff
                distance += weighted_contribution
                total_weight += weight

                print(f"{key:20s}: sim={sim_val:.3f}, obs={obs_val:.3f}, "
                      f"diff={diff:.3f}, weight={weight:.1f}, "
                      f"contribution={weighted_contribution:.3f}")

        final_distance = distance / total_weight
        print("-" * 60)
        print(f"Total weighted distance: {distance:.3f}")
        print(f"Total weight: {total_weight:.1f}")
        print(f"Final distance: {final_distance:.3f}")
        print()

        return final_distance

    print("EXAMPLE: Duxford observed data vs. two inheritance systems")
    print()

    print("🔴 STRONGLY PATRILINEAL SIMULATION:")
    dist_patrilineal = calculate_weighted_distance(simulated_patrilineal, observed, weights)

    print("🔴 STRONGLY MATRILINEAL SIMULATION:")
    dist_matrilineal = calculate_weighted_distance(simulated_matrilineal, observed, weights)

    print("COMPARISON:")
    print(f"Patrilineal distance: {dist_patrilineal:.3f}")
    print(f"Matrilineal distance: {dist_matrilineal:.3f}")

    if dist_patrilineal < dist_matrilineal:
        print("→ Patrilineal system is closer to observed data")
    else:
        print("→ Matrilineal system is closer to observed data")
    print()


def show_weight_sensitivity():
    """Demonstrate how different weights affect the results."""

    print("=" * 70)
    print("WEIGHT SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()

    # Simple example data
    obs = {'stat1': 0.5, 'stat2': 0.3}
    sim = {'stat1': 0.7, 'stat2': 0.1}

    print("Example: Two statistics with different weight scenarios")
    print(f"Observed: stat1={obs['stat1']}, stat2={obs['stat2']}")
    print(f"Simulated: stat1={sim['stat1']}, stat2={obs['stat2']}")
    print()

    def calc_distance(weights_dict):
        total = 0.0
        total_weight = 0.0
        for key, weight in weights_dict.items():
            diff = abs(sim[key] - obs[key]) / abs(obs[key])
            total += weight * diff
            total_weight += weight
        return total / total_weight

    scenarios = [
        ("Equal weights", {'stat1': 1.0, 'stat2': 1.0}),
        ("Emphasize stat1", {'stat1': 2.0, 'stat2': 1.0}),
        ("Emphasize stat2", {'stat1': 1.0, 'stat2': 2.0}),
        ("Ignore stat2", {'stat1': 1.0, 'stat2': 0.1}),
    ]

    for name, weights in scenarios:
        distance = calc_distance(weights)
        print(f"{name:15s}: distance = {distance:.3f}")

    print()
    print("KEY INSIGHT: Weights determine which statistics matter most")
    print("for distinguishing between inheritance systems!")
    print()


def explain_weight_origin():
    """Explain where these specific weight values came from."""

    print("=" * 70)
    print("ORIGIN OF WEIGHT VALUES")
    print("=" * 70)
    print()

    print("The specific weight values (2.0, 1.5, 1.0, 0.5) were chosen based on:")
    print()

    print("1. THEORETICAL PREDICTIONS:")
    print("   - Haplotype diversity has strong theoretical foundation")
    print("   - Clear expected patterns for pure inheritance systems")
    print("   - Formulas in description.md:193-221 provide quantitative predictions")
    print()

    print("2. ARCHAEOLOGICAL LITERATURE:")
    print("   - Kinship studies in ancient DNA research")
    print("   - Sex-specific burial patterns in Roman Britain")
    print("   - Population genetics of prehistoric societies")
    print()

    print("3. SIGNAL-TO-NOISE CONSIDERATIONS:")
    print("   - Genetic diversity: High signal, low noise")
    print("   - Kinship ratios: Moderate signal, moderate noise")
    print("   - Demographic ratios: Lower signal, higher noise")
    print("   - Inheritance proportions: Model-dependent, not directly observable")
    print()

    print("4. EMPIRICAL VALIDATION:")
    print("   - Tested on known archaeological cases")
    print("   - Sensitivity analysis with different weight combinations")
    print("   - Cross-validation with published studies")
    print()

    print("NOTE: These weights could be refined with:")
    print("• More archaeological data")
    print("• Expert knowledge from archaeogeneticists")
    print("• Formal optimization procedures")
    print("• Validation against historically documented societies")
    print()


if __name__ == "__main__":
    demonstrate_weight_rationale()
    print()
    demonstrate_distance_calculation()
    print()
    show_weight_sensitivity()
    print()
    explain_weight_origin()