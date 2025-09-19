"""
Demonstration of differences between Strong and Weak Patrilineal inheritance systems.

Shows exactly what parameters change and their expected effects on the population.
"""

import numpy as np
import pandas as pd
from agent_simulation import SimulationParameters, InheritanceSimulator
import matplotlib.pyplot as plt


def compare_inheritance_probabilities():
    """Show the key difference: inheritance probability by sex."""

    print("=" * 70)
    print("INHERITANCE PROBABILITY DIFFERENCES")
    print("=" * 70)
    print()

    systems = {
        "strongly_patrilineal": {"male": 0.9, "female": 0.1},
        "weakly_patrilineal": {"male": 0.7, "female": 0.3},
        "balanced": {"male": 0.5, "female": 0.5},
        "weakly_matrilineal": {"male": 0.3, "female": 0.7},
        "strongly_matrilineal": {"male": 0.1, "female": 0.9}
    }

    print("System                | Male Inheritance | Female Inheritance | Difference")
    print("-" * 70)

    for system, probs in systems.items():
        male_prob = probs["male"]
        female_prob = probs["female"]
        difference = male_prob - female_prob

        print(f"{system:20s} | {male_prob:15.1f} | {female_prob:17.1f} | {difference:+9.1f}")

    print()
    print("KEY INSIGHT: Only inheritance probabilities change between systems!")
    print("All other parameters (burial rates, aDNA success, migration) stay the same.")
    print()


def demonstrate_inheritance_effects():
    """Show how inheritance probability affects who stays vs. who leaves."""

    print("=" * 70)
    print("INHERITANCE EFFECTS ON POPULATION COMPOSITION")
    print("=" * 70)
    print()

    # Simulate inheritance decisions for 100 individuals
    np.random.seed(42)  # For reproducible results

    n_individuals = 100
    males = 50
    females = 50

    systems = ["strongly_patrilineal", "weakly_patrilineal"]

    for system in systems:
        print(f"🔹 {system.upper()}:")

        if system == "strongly_patrilineal":
            male_inherit_prob = 0.9
            female_inherit_prob = 0.1
        else:  # weakly_patrilineal
            male_inherit_prob = 0.7
            female_inherit_prob = 0.3

        # Simulate inheritance decisions
        male_inheritors = np.sum(np.random.random(males) < male_inherit_prob)
        female_inheritors = np.sum(np.random.random(females) < female_inherit_prob)

        total_inheritors = male_inheritors + female_inheritors

        print(f"  Male inheritors:   {male_inheritors:2d}/{males} ({male_inheritors/males:5.1%})")
        print(f"  Female inheritors: {female_inheritors:2d}/{females} ({female_inheritors/females:5.1%})")
        print(f"  Total inheritors:  {total_inheritors:2d}/{n_individuals} ({total_inheritors/n_individuals:5.1%})")
        print(f"  Male/Female ratio: {male_inheritors/female_inheritors if female_inheritors > 0 else 'inf':.1f}")
        print()

    print("EFFECT: Inheritance status determines:")
    print("• Who stays in the community (inheritors)")
    print("• Who is more likely to be buried locally")
    print("• Who passes on their lineage at the site")
    print()


def show_burial_cascade_effects():
    """Show how inheritance affects burial probability and final composition."""

    print("=" * 70)
    print("CASCADE EFFECTS: INHERITANCE → BURIAL → aDNA → FINAL SAMPLE")
    print("=" * 70)
    print()

    base_burial_prob = 0.8
    inheritor_burial_multiplier = 1.5
    adna_success_rate = 0.7

    print("BURIAL PROBABILITY CALCULATION:")
    print(f"• Base burial probability: {base_burial_prob}")
    print(f"• Inheritor multiplier: {inheritor_burial_multiplier}x")
    print(f"• aDNA success rate: {adna_success_rate}")
    print()

    # Calculate effective burial rates
    non_inheritor_burial = base_burial_prob
    inheritor_burial = min(base_burial_prob * inheritor_burial_multiplier, 1.0)

    print("EFFECTIVE BURIAL RATES:")
    print(f"• Non-inheritors: {non_inheritor_burial:.1%}")
    print(f"• Inheritors: {inheritor_burial:.1%}")
    print()

    # Expected composition for different systems
    systems = {
        "strongly_patrilineal": {"m_inherit": 0.9, "f_inherit": 0.1},
        "weakly_patrilineal": {"m_inherit": 0.7, "f_inherit": 0.3}
    }

    print("EXPECTED FINAL COMPOSITION (per 100 individuals, 50M/50F):")
    print("-" * 60)

    for system, probs in systems.items():
        print(f"\n🔹 {system.upper()}:")

        # Calculate expected buried individuals
        m_inheritors = 50 * probs["m_inherit"]
        f_inheritors = 50 * probs["f_inherit"]
        m_non_inheritors = 50 - m_inheritors
        f_non_inheritors = 50 - f_inheritors

        # Expected burials
        m_buried = (m_inheritors * inheritor_burial +
                   m_non_inheritors * non_inheritor_burial)
        f_buried = (f_inheritors * inheritor_burial +
                   f_non_inheritors * non_inheritor_burial)

        # Expected aDNA success
        m_adna = m_buried * adna_success_rate
        f_adna = f_buried * adna_success_rate

        print(f"  Expected buried:     {m_buried:.1f}M, {f_buried:.1f}F")
        print(f"  Expected with aDNA:  {m_adna:.1f}M, {f_adna:.1f}F")
        print(f"  Final sex ratio:     {m_adna/f_adna:.2f}")
        print(f"  Male proportion:     {m_adna/(m_adna+f_adna):.1%}")


def show_kinship_pattern_differences():
    """Show how inheritance affects kinship relationship patterns."""

    print("\n" + "=" * 70)
    print("KINSHIP PATTERN DIFFERENCES")
    print("=" * 70)
    print()

    print("EXPECTED KINSHIP PATTERNS:")
    print()

    print("🔴 STRONGLY PATRILINEAL (90% male inheritance):")
    print("  • Many father-son relationships (both inherit)")
    print("  • Few mother-daughter relationships (daughters rarely inherit)")
    print("  • Male-male kinship dominates")
    print("  • Females less likely to be buried together")
    print("  • Expected father-son ratio: HIGH (~40-60%)")
    print("  • Expected mother-daughter ratio: LOW (~5-15%)")
    print()

    print("🟡 WEAKLY PATRILINEAL (70% male inheritance):")
    print("  • Moderate father-son relationships")
    print("  • Some mother-daughter relationships (30% chance)")
    print("  • Mixed kinship patterns")
    print("  • More balanced but still male-biased")
    print("  • Expected father-son ratio: MODERATE (~25-40%)")
    print("  • Expected mother-daughter ratio: MODERATE (~15-30%)")
    print()

    print("KEY DIFFERENCES:")
    print("1. DEGREE of sex bias in inheritance")
    print("2. PROPORTION of same-sex relationships")
    print("3. BALANCE between male and female kinship")
    print("4. EXPECTED statistical signatures")


def demonstrate_with_simulation():
    """Run actual simulations to show the differences."""

    print("\n" + "=" * 70)
    print("ACTUAL SIMULATION COMPARISON")
    print("=" * 70)
    print()

    # Run simulations for both systems
    systems = ["strongly_patrilineal", "weakly_patrilineal"]
    results = {}

    for system in systems:
        print(f"Running {system} simulation...")

        params = SimulationParameters(
            inheritance_system=system,
            generations=3,
            population_per_generation=20
        )

        simulator = InheritanceSimulator(params)
        result = simulator.run_simulation()
        results[system] = result

        # Extract key statistics
        stats = result['statistics']
        kinship_pairs = result['kinship_pairs']

        print(f"  Total population: {result['total_population']}")
        print(f"  Buried individuals: {result['buried_count']}")
        print(f"  aDNA successful: {result['adna_count']}")
        print(f"  Y-chromosome diversity: {stats['y_diversity']:.3f}")
        print(f"  mtDNA diversity: {stats['mt_diversity']:.3f}")
        print(f"  Kinship pairs found: {len(kinship_pairs)}")

        # Analyze kinship patterns
        if kinship_pairs:
            kinship_df = pd.DataFrame(kinship_pairs)

            father_son = len(kinship_df[
                (kinship_df['Relationship'].str.contains('Parent-Offspring', na=False)) &
                (kinship_df['Sex_1'] == 'M') & (kinship_df['Sex_2'] == 'M')
            ])

            mother_daughter = len(kinship_df[
                (kinship_df['Relationship'].str.contains('Parent-Offspring', na=False)) &
                (kinship_df['Sex_1'] == 'F') & (kinship_df['Sex_2'] == 'F')
            ])

            total_kinship = len(kinship_pairs)

            if total_kinship > 0:
                print(f"  Father-son pairs: {father_son} ({father_son/total_kinship:.1%})")
                print(f"  Mother-daughter pairs: {mother_daughter} ({mother_daughter/total_kinship:.1%})")

        print()

    print("COMPARISON:")
    print(f"Strong vs Weak Patrilineal differences:")
    print(f"• Y-diversity: {results['strongly_patrilineal']['statistics']['y_diversity']:.3f} vs {results['weakly_patrilineal']['statistics']['y_diversity']:.3f}")
    print(f"• mtDNA diversity: {results['strongly_patrilineal']['statistics']['mt_diversity']:.3f} vs {results['weakly_patrilineal']['statistics']['mt_diversity']:.3f}")
    print()
    print("NOTE: Differences may be subtle in small simulations.")
    print("Patterns become clearer with larger samples and multiple runs.")


if __name__ == "__main__":
    compare_inheritance_probabilities()
    demonstrate_inheritance_effects()
    show_burial_cascade_effects()
    show_kinship_pattern_differences()
    demonstrate_with_simulation()

    print("\n" + "=" * 70)
    print("SUMMARY: THE ONLY DIFFERENCE IS INHERITANCE PROBABILITY")
    print("=" * 70)
    print()
    print("Strong Patrilineal: 90% males inherit, 10% females inherit")
    print("Weak Patrilineal:   70% males inherit, 30% females inherit")
    print()
    print("This single change cascades through:")
    print("• Who stays in the community")
    print("• Who gets buried locally")
    print("• What kinship patterns emerge")
    print("• Final genetic diversity patterns")
    print()
    print("The 'strength' refers to how extreme the sex bias is!")