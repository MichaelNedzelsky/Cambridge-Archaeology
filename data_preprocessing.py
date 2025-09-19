"""
Data preprocessing for Cambridge Archaeology aDNA analysis.

Prepares site-specific data for agent-based model testing of inheritance systems.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class SiteDataProcessor:
    """Process archaeological site data for inheritance pattern analysis."""

    def __init__(self, data_file: str, kinship_file: Optional[str] = None):
        """
        Initialize with data files.

        Args:
            data_file: Path to combined CSV with individual data
            kinship_file: Optional path to kinship relationships CSV
        """
        self.data_df = pd.read_csv(data_file)
        self.kinship_df = None

        if kinship_file:
            # The kinship CSV has multi-row headers, skip first row
            self.kinship_df = pd.read_csv(kinship_file, skiprows=1)
            self.kinship_df.columns = [
                'Site', 'Individual_1', 'Individual_2', 'Y_chr_1', 'Y_chr_2', 'Y_identical',
                'mtDNA_1', 'mtDNA_2', 'mt_identical', 'blank', 'Degree', 'Predicted', 'Likely_relationship'
            ]

        self._standardize_columns()

    def _standardize_columns(self):
        """Standardize column names for consistent access."""
        # Create mapping of lowercase column names to actual names
        self.col_map = {col.lower().strip().replace(' ', '_').replace('-', '_'): col
                       for col in self.data_df.columns}

        # Essential columns
        self.site_col = self._get_column(['site', 'site_group', 'site_id'])
        self.sex_col = self._get_column(['sex'])
        self.age_col = self._get_column(['age', 'site_age_(ce)'])
        self.y_chr_col = self._get_column(['y_chr_haplogroup', 'y_chromosome'])
        self.mt_dna_col = self._get_column(['mtdna_haplogroup', 'mitochondrial_dna'])
        self.adult_col = self._get_column(['adult', 'adult_(y/n/unknown)'])
        self.sample_id_col = self._get_column(['sample_id'])

    def _get_column(self, possible_names: List[str]) -> Optional[str]:
        """Find the actual column name from possible variations."""
        for name in possible_names:
            name_normalized = name.lower().replace(' ', '_').replace('-', '_')
            if name_normalized in self.col_map:
                return self.col_map[name_normalized]
        return None

    def get_sites(self) -> List[str]:
        """Get list of unique sites in the dataset."""
        if not self.site_col:
            raise ValueError("No site column found in data")
        return sorted(self.data_df[self.site_col].dropna().unique())

    def get_site_data(self, site_name: str) -> pd.DataFrame:
        """Get all data for a specific site."""
        if not self.site_col:
            raise ValueError("No site column found in data")

        site_data = self.data_df[self.data_df[self.site_col] == site_name].copy()

        # Clean and standardize data
        site_data = self._clean_site_data(site_data)

        return site_data

    def _clean_site_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize site data."""
        df = df.copy()

        # Standardize sex column
        if self.sex_col:
            df['sex_clean'] = df[self.sex_col].astype(str).str.upper()
            df['sex_clean'] = df['sex_clean'].str.extract(r'([MF])', expand=False)
            df['sex_clean'] = df['sex_clean'].fillna('Unknown')

        # Standardize age/adult status
        if self.adult_col:
            df['is_adult'] = df[self.adult_col].astype(str).str.upper().str.startswith('Y')
        elif self.age_col:
            df['is_adult'] = df[self.age_col].astype(str).str.upper().str.contains('ADULT')
        else:
            df['is_adult'] = True  # Default assumption

        # Clean haplogroup data
        if self.y_chr_col:
            df['y_chr_clean'] = self._clean_haplogroup(df[self.y_chr_col])

        if self.mt_dna_col:
            df['mt_dna_clean'] = self._clean_haplogroup(df[self.mt_dna_col])

        return df

    def _clean_haplogroup(self, series: pd.Series) -> pd.Series:
        """Clean haplogroup data by removing low coverage and standardizing."""
        cleaned = series.astype(str)

        # Mark invalid/missing data
        invalid_patterns = [
            'too low coverage', 'unknown', 'nan', 'untested',
            '#n/a', 'n/a', '', 'none'
        ]

        for pattern in invalid_patterns:
            cleaned = cleaned.str.replace(pattern, '', case=False)

        # Remove everything after dash (as mentioned in description.md:47)
        cleaned = cleaned.str.split('-').str[0]

        # Replace empty strings with NaN
        cleaned = cleaned.replace('', np.nan)

        return cleaned

    def get_site_kinship(self, site_name: str) -> pd.DataFrame:
        """Get kinship data for a specific site."""
        if self.kinship_df is None:
            return pd.DataFrame()

        site_kinship = self.kinship_df[
            self.kinship_df['Site'].str.lower() == site_name.lower()
        ].copy()

        return site_kinship

    def calculate_site_statistics(self, site_name: str) -> Dict:
        """Calculate comprehensive statistics for a site."""
        site_data = self.get_site_data(site_name)
        site_kinship = self.get_site_kinship(site_name)

        stats = {
            'site_name': site_name,
            'total_individuals': len(site_data),
            'adults': len(site_data[site_data['is_adult']]),
            'non_adults': len(site_data[~site_data['is_adult']]),
            'males': len(site_data[site_data['sex_clean'] == 'M']),
            'females': len(site_data[site_data['sex_clean'] == 'F']),
            'sex_unknown': len(site_data[site_data['sex_clean'] == 'Unknown']),
        }

        # Haplogroup statistics
        if 'y_chr_clean' in site_data.columns:
            y_valid = site_data['y_chr_clean'].dropna()
            stats['y_chr_samples'] = len(y_valid)
            stats['y_chr_diversity'] = self._calculate_nei_diversity(y_valid)
            stats['y_chr_unique'] = len(y_valid.unique())

        if 'mt_dna_clean' in site_data.columns:
            mt_valid = site_data['mt_dna_clean'].dropna()
            stats['mt_dna_samples'] = len(mt_valid)
            stats['mt_dna_diversity'] = self._calculate_nei_diversity(mt_valid)
            stats['mt_dna_unique'] = len(mt_valid.unique())

        # Kinship statistics
        if not site_kinship.empty:
            stats['kinship_pairs'] = len(site_kinship)
            stats['first_degree'] = len(site_kinship[site_kinship['Degree'].str.contains('First', na=False)])
            stats['second_degree'] = len(site_kinship[site_kinship['Degree'].str.contains('Second', na=False)])
            stats['third_degree'] = len(site_kinship[site_kinship['Degree'].str.contains('Third', na=False)])

            # Relationship type analysis
            stats.update(self._analyze_kinship_types(site_kinship))

        return stats

    def _calculate_nei_diversity(self, haplogroups: pd.Series) -> float:
        """Calculate Nei's haplotype diversity."""
        if len(haplogroups) <= 1:
            return 0.0

        counts = haplogroups.value_counts()
        n = len(haplogroups)

        # Calculate sum of squared frequencies
        freq_sum_sq = sum((count / n) ** 2 for count in counts)

        # Nei's formula: H = (n/(n-1)) * (1 - sum(p_i^2))
        diversity = (n / (n - 1)) * (1 - freq_sum_sq)

        return diversity

    def _analyze_kinship_types(self, kinship_df: pd.DataFrame) -> Dict:
        """Analyze types of kinship relationships."""
        relationship_stats = defaultdict(int)

        for _, row in kinship_df.iterrows():
            rel_text = str(row['Likely_relationship']).lower()

            # Categorize relationships
            if 'father' in rel_text and 'son' in rel_text:
                relationship_stats['father_son'] += 1
            elif 'mother' in rel_text and ('son' in rel_text or 'daughter' in rel_text):
                if 'son' in rel_text:
                    relationship_stats['mother_son'] += 1
                else:
                    relationship_stats['mother_daughter'] += 1
            elif 'sister' in rel_text or 'brother' in rel_text:
                if 'sister' in rel_text and 'brother' not in rel_text:
                    relationship_stats['sisters'] += 1
                elif 'brother' in rel_text and 'sister' not in rel_text:
                    relationship_stats['brothers'] += 1
                else:
                    relationship_stats['siblings'] += 1

        return dict(relationship_stats)

    def get_all_sites_summary(self) -> pd.DataFrame:
        """Get summary statistics for all sites."""
        sites = self.get_sites()
        summaries = []

        for site in sites:
            try:
                stats = self.calculate_site_statistics(site)
                summaries.append(stats)
            except Exception as e:
                print(f"Error processing site {site}: {e}")
                continue

        return pd.DataFrame(summaries)


def load_and_preprocess_data(data_file: str = 'cleaned_dataset.csv',
                           kinship_file: str = 'Cambridshire aDNA summary data.xlsx - DNA kinship details.csv') -> SiteDataProcessor:
    """Convenience function to load and preprocess the data."""
    return SiteDataProcessor(data_file, kinship_file)


if __name__ == "__main__":
    # Test the preprocessing
    processor = load_and_preprocess_data()

    print("Sites found:", processor.get_sites())
    print("\nSummary statistics:")

    summary_df = processor.get_all_sites_summary()
    print(summary_df.to_string(index=False))

    # Example: detailed analysis for Duxford
    print("\n" + "="*50)
    print("DETAILED ANALYSIS: DUXFORD")
    print("="*50)

    duxford_data = processor.get_site_data('Duxford')
    duxford_kinship = processor.get_site_kinship('Duxford')

    print(f"Total individuals: {len(duxford_data)}")
    print(f"Kinship pairs: {len(duxford_kinship)}")

    if not duxford_kinship.empty:
        print("\nKinship relationships:")
        for _, row in duxford_kinship.iterrows():
            print(f"  {row['Individual_1']} - {row['Individual_2']}: {row['Likely_relationship']}")