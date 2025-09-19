"""
Global parameters for the Cambridge Archaeology inheritance pattern analysis.

These parameters apply across all sites and simulations.
"""

# Burial and preservation parameters
BURIAL_PROBABILITY = 0.8  # Probability that an individual is buried in the cemetery
ADNA_SUCCESS_RATE = 0.7   # Probability that aDNA can be successfully extracted

# Simulation parameters
MATING_PROBABILITY = 0.8   # Probability that paired individuals produce offspring

# Simulation parameters
DEFAULT_GENERATIONS = 4    # Number of generations to simulate (cemetery timespan)

# Haplogroup pools for initialization
# Y-chromosome haplogroups common in Roman Britain
DEFAULT_Y_HAPLOGROUPS = ['R1b', 'I2', 'G2a', 'R1a', 'I1']

# Mitochondrial haplogroups common in Roman Britain
DEFAULT_MT_HAPLOGROUPS = ['H', 'U5', 'K1', 'J1', 'T2', 'W', 'V', 'U4', 'H1', 'H3']

# ABC parameters
DEFAULT_SIMULATIONS_PER_SYSTEM = 100  # Number of simulations per inheritance system
ABC_ACCEPTANCE_QUANTILE = 0.05        # Top 5% of simulations accepted in ABC

# Distance calculation weights for ABC
# Higher weights = more importance in determining inheritance system
DISTANCE_WEIGHTS = {
    'y_diversity': 2.0,           # High - primary genetic discriminator
    'mt_diversity': 2.0,          # High - complementary genetic signal
    'prop_father_son': 1.5,       # Medium - direct kinship evidence
    'prop_mother_daughter': 1.5,  # Medium - direct kinship evidence
    'sex_ratio': 1.0,             # Standard - demographic information
    'prop_y_matches': 1.0,        # Standard - quality control
    'prop_mt_matches': 1.0,       # Standard - quality control
    'prop_inheritors': 0.5,       # Low - model artifact
}

# Inheritance system definitions
# Only female_prob is stored; male_prob = 1 - female_prob
INHERITANCE_SYSTEMS = {
    'strongly_patrilineal': {
        'female_prob': 0.1,
        'description': '90% male, 10% female inheritance'
    },
    'weakly_patrilineal': {
        'female_prob': 0.3,
        'description': '70% male, 30% female inheritance'
    },
    'balanced': {
        'female_prob': 0.5,
        'description': '50% male, 50% female inheritance'
    },
    'weakly_matrilineal': {
        'female_prob': 0.7,
        'description': '30% male, 70% female inheritance'
    },
    'strongly_matrilineal': {
        'female_prob': 0.9,
        'description': '10% male, 90% female inheritance'
    }
}

def get_inheritance_probabilities(system_name):
    """
    Get male and female inheritance probabilities for a given system.

    Parameters:
    -----------
    system_name : str
        Name of the inheritance system

    Returns:
    --------
    tuple : (male_probability, female_probability)
    """
    if system_name not in INHERITANCE_SYSTEMS:
        raise ValueError(f"Unknown inheritance system: {system_name}")

    system = INHERITANCE_SYSTEMS[system_name]
    female_prob = system['female_prob']
    male_prob = 1.0 - female_prob
    return male_prob, female_prob

def get_system_description(system_name):
    """
    Get human-readable description of an inheritance system.

    Parameters:
    -----------
    system_name : str
        Name of the inheritance system

    Returns:
    --------
    str : Description of the system
    """
    if system_name not in INHERITANCE_SYSTEMS:
        raise ValueError(f"Unknown inheritance system: {system_name}")

    return INHERITANCE_SYSTEMS[system_name]['description']

if __name__ == "__main__":
    # Display global parameters when run directly
    print("Cambridge Archaeology Global Parameters")
    print("=" * 50)

    print("\nBurial and Preservation:")
    print(f"  Burial probability: {BURIAL_PROBABILITY:.0%}")
    print(f"  aDNA success rate: {ADNA_SUCCESS_RATE:.0%}")
    print(f"  Mating probability: {MATING_PROBABILITY:.0%}")

    print("\nSimulation Settings:")
    print(f"  Generations simulated: {DEFAULT_GENERATIONS}")
    print(f"  Simulations per system: {DEFAULT_SIMULATIONS_PER_SYSTEM}")
    print(f"  ABC acceptance rate: {ABC_ACCEPTANCE_QUANTILE:.0%}")

    print("\nInheritance Systems:")
    for system_name, system_info in INHERITANCE_SYSTEMS.items():
        print(f"  {system_name}: {system_info['description']}")

    print("\nABC Distance Weights:")
    for metric, weight in DISTANCE_WEIGHTS.items():
        print(f"  {metric}: {weight}")