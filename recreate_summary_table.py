import pandas as pd
import numpy as np

def clean_sex_data(sex_series):
    """Standardizes the sex data to 'M', 'F', or None."""
    # Replace bracketed values like '(M)' with 'M'
    sex_series = sex_series.str.replace(r'[\(\)]', '', regex=True)
    # Map standardized values
    sex_series = sex_series.str.upper().str.strip()
    return sex_series.map({'M': 'M', 'F': 'F'}).fillna(np.nan)

def calculate_summary_table(data_file, kinship_file):
    """
    Generates a summary table from archaeological and kinship data.

    Args:
        data_file (str): Path to the main data CSV (e.g., combined_grouped.csv).
        kinship_file (str): Path to the kinship details CSV.
    
    Returns:
        pandas.DataFrame: A dataframe containing the summary statistics for each site.
    """
    try:
        # Load the main dataset
        df = pd.read_csv(data_file)
        # Load the kinship dataset, skipping the first row which acts as a sub-header
        kinship_df = pd.read_csv(kinship_file, header=1)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        return None

    # --- Data Cleaning ---
    df['Sex'] = clean_sex_data(df['Sex'].astype(str))
    kinship_df.rename(columns={'Degree': 'Kinship_Degree'}, inplace=True)
    kinship_df['Kinship_Degree'] = kinship_df['Kinship_Degree'].str.strip()

    # Get a list of unique sites to iterate over
    sites = df['Site Group'].unique()
    summary_data = []

    # --- Analysis Loop for each site ---
    for site in sorted(sites):
        site_df = df[df['Site Group'] == site]
        site_kinship_df = kinship_df[kinship_df['Site'].str.contains(site, case=False, na=False)]

        # Filter for individuals with successful DNA tests
        dna_tested_df = site_df[site_df['DNA tested'] == 'Y']
        males = dna_tested_df[dna_tested_df['Sex'] == 'M']
        females = dna_tested_df[dna_tested_df['Sex'] == 'F']

        # Calculate Y-DNA Diversity
        y_haplotypes = males['Y-chr Haplogroup'].dropna()
        y_diversity = 0
        if not y_haplotypes.empty:
            y_diversity = y_haplotypes.nunique() / len(y_haplotypes)

        # Calculate mtDNA Diversity
        mtdna_haplotypes = females['mtDNA Haplogroup'].dropna()
        mtdna_diversity = 0
        if not mtdna_haplotypes.empty:
            mtdna_diversity = mtdna_haplotypes.nunique() / len(mtdna_haplotypes)

        # Count Kinship Pairs
        first_degree_pairs = site_kinship_df[site_kinship_df['Kinship_Degree'] == 'First Degree'].shape[0]
        second_degree_pairs = site_kinship_df[site_kinship_df['Kinship_Degree'] == 'Second Degree'].shape[0]

        summary_data.append({
            'Site': site,
            'n Individuals': len(site_df),
            'n DNA-tested': len(dna_tested_df),
            'n Male (DNA)': len(males),
            'n Female (DNA)': len(females),
            'Y-DNA Diversity (H)': f"{y_diversity:.2f}",
            'mtDNA Diversity (H)': f"{mtdna_diversity:.2f}",
            '1st Degree Pairs': first_degree_pairs,
            '2nd Degree Pairs': second_degree_pairs,
        })

    return pd.DataFrame(summary_data)


if __name__ == '__main__':
    # Define the paths to your data files
    main_data_filepath = 'combined_grouped.csv'
    kinship_data_filepath = 'Cambridshire aDNA summary data.xlsx - DNA kinship details.csv'

    # Generate the summary table
    summary_table = calculate_summary_table(main_data_filepath, kinship_data_filepath)

    if summary_table is not None:
        print("--- Summary of Genetic and Kinship Data from Cambridgeshire Sites (Table 2) ---")
        # Print the dataframe without the index column for a cleaner look
        print(summary_table.to_string(index=False))
