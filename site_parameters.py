"""
Site-specific parameters for Roman-era Cambridgeshire cemeteries.

Population estimates are per generation averages based on archaeological evidence.
Date ranges in CE (Common Era).
"""

SITE_PARAMETERS = {
    'Duxford': {
        'total_burials': 36,
        'population_per_generation': 25,  # 20-30 average
        'date_range': (100, 125),  # 1st-early 2nd century AD
        'status': 'low',
        'description': 'Small to mid-sized farmstead from late prehistory into Roman period',
        'notes': 'Most likely site for matrilocality due to early date and minimal Roman influence'
    },

    'NW_Cambridge': {
        'total_burials': 14,
        'population_per_generation': 25,  # 20-30 average
        'date_range': (150, 250),
        'status': 'low-medium',
        'description': 'Small cemetery associated with mid-sized farm',
        'notes': 'One of several cemeteries for the settlement'
    },

    'Vicar_Farm': {
        'total_burials': 29,
        'population_per_generation': 40,  # 30-50 average
        'date_range': (270, 420),  # late 3rd-4th century AD
        'status': 'medium-high',
        'description': 'Large Roman farm in northwest Cambridge',
        'notes': 'Higher status than regular farms but below villa status'
    },

    'Fenstanton': {
        'total_burials': 48,
        'population_per_generation': 30,  # up to 30
        'date_range': (40, 400),  # 1st-4th centuries AD
        'status': 'medium',
        'description': 'Large farm by Roman road with specialist cattle butchery',
        'notes': 'Evidence of domestic, craft, and small-scale industrial activity'
    },

    'Knobbs': {
        'total_burials': 56,
        'population_per_generation': 40,  # 30-50 average
        'date_range': (275, 400),  # mid-4th century AD
        'status': 'medium',
        'description': 'Three small cemeteries at edge of large farm',
        'notes': 'Supplied grain and animals to nearby village and trading port'
    }
}

# Note: Arbury is excluded from analysis as per data cleaning process
# It was a high-status villa mausoleum with 6 burials in stone coffins

def get_site_population(site_name):
    """
    Get the population per generation for a specific site.

    Parameters:
    -----------
    site_name : str
        Name of the site (must match keys in SITE_PARAMETERS)

    Returns:
    --------
    int : Population per generation for the site
    """
    if site_name not in SITE_PARAMETERS:
        raise ValueError(f"Unknown site: {site_name}. Valid sites are: {list(SITE_PARAMETERS.keys())}")
    return SITE_PARAMETERS[site_name]['population_per_generation']

def get_all_site_populations():
    """
    Get a dictionary of all sites and their population per generation.

    Returns:
    --------
    dict : Site names as keys, population per generation as values
    """
    return {site: params['population_per_generation']
            for site, params in SITE_PARAMETERS.items()}

def get_site_info(site_name):
    """
    Get complete information for a specific site.

    Parameters:
    -----------
    site_name : str
        Name of the site

    Returns:
    --------
    dict : Complete site parameters
    """
    if site_name not in SITE_PARAMETERS:
        raise ValueError(f"Unknown site: {site_name}")
    return SITE_PARAMETERS[site_name].copy()

# For backward compatibility with existing simulation code
DEFAULT_POPULATION_SIZE = 30  # Average across all sites

if __name__ == "__main__":
    # Display site information when run directly
    print("Roman-era Cambridgeshire Site Parameters")
    print("=" * 50)

    for site, params in SITE_PARAMETERS.items():
        print(f"\n{site}:")
        print(f"  Burials: {params['total_burials']}")
        print(f"  Population/generation: {params['population_per_generation']}")
        print(f"  Date range: {params['date_range'][0]}-{params['date_range'][1]} CE")
        print(f"  Status: {params['status']}")
        print(f"  Description: {params['description']}")