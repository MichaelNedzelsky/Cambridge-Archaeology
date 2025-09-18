"""
Statistical measures for detecting inheritance patterns in archaeological aDNA data.

Implements theoretical framework from description.md for distinguishing between:
- Strongly matrilineal
- Weakly matrilineal
- Balanced
- Weakly patrilineal
- Strongly patrilineal

Based on theoretical results in description.md:193-221
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from scipy import stats
from data_preprocessing import SiteDataProcessor


class InheritancePatternAnalyzer:
    """Analyze inheritance patterns using multiple statistical measures."""

    def __init__(self, processor: SiteDataProcessor):
        self.processor = processor

    def calculate_nei_diversity(self, haplogroups: pd.Series) -> float:
        """
        Calculate Nei's haplotype diversity.
        H = (n/(n-1)) * (1 - sum(p_i^2)) for n > 1, else 0
        """
        valid_haplos = haplogroups.dropna()
        n = len(valid_haplos)

        if n <= 1:
            return 0.0

        counts = valid_haplos.value_counts()
        freq_sum_sq = sum((count / n) ** 2 for count in counts)

        return (n / (n - 1)) * (1 - freq_sum_sq)

    def calculate_theoretical_diversity(self, generations: int, system: str) -> Dict[str, float]:
        """
        Calculate theoretical haplotype diversity based on inheritance system.

        From description.md formulas:
        - Pure patrilineal: h_Y = 0, h_mt = 2g(g-1)+2) / (g(2g-1))
        - Pure matrilineal: h_Y = 1, h_mt = 2g / ((2g-1)^2)
        """
        g = generations

        if system == "strongly_patrilineal":
            # Pure patrilineal descent
            h_y = 0.0
            h_mt = (2 * g * (g - 1) + 2) / (g * (2 * g - 1)) if g > 1 else 0.0

        elif system == "strongly_matrilineal":
            # Pure matrilineal descent
            h_y = 1.0 if g > 0 else 0.0
            h_mt = (2 * g) / ((2 * g - 1) ** 2) if g > 1 else 0.0

        else:
            # Mixed systems - interpolate between extremes
            if system == "weakly_patrilineal":
                patrilineal_weight = 0.7
            elif system == "weakly_matrilineal":
                patrilineal_weight = 0.3
            else:  # balanced
                patrilineal_weight = 0.5

            # Interpolate between pure patrilineal and matrilineal
            pat_h_y, pat_h_mt = 0.0, (2 * g * (g - 1) + 2) / (g * (2 * g - 1)) if g > 1 else 0.0
            mat_h_y, mat_h_mt = 1.0 if g > 0 else 0.0, (2 * g) / ((2 * g - 1) ** 2) if g > 1 else 0.0

            h_y = patrilineal_weight * pat_h_y + (1 - patrilineal_weight) * mat_h_y
            h_mt = patrilineal_weight * pat_h_mt + (1 - patrilineal_weight) * mat_h_mt

        return {"h_y": h_y, "h_mt": h_mt}

    def calculate_kinship_ratios(self, site_name: str) -> Dict[str, float]:
        """Calculate kinship relationship ratios that distinguish inheritance systems."""
        kinship_df = self.processor.get_site_kinship(site_name)

        if kinship_df.empty:
            return {
                "father_son_ratio": 0.0,
                "mother_daughter_ratio": 0.0,
                "same_sex_ratio": 0.0,
                "cross_sex_ratio": 0.0,
                "male_kinship_ratio": 0.0,
                "female_kinship_ratio": 0.0
            }

        # Count relationship types
        relationships = {
            "father_son": 0,
            "mother_son": 0,
            "mother_daughter": 0,
            "sisters": 0,
            "brothers": 0,
            "siblings": 0,
        }

        total_relationships = len(kinship_df)

        for _, row in kinship_df.iterrows():
            rel_text = str(row['Likely_relationship']).lower()

            if 'father' in rel_text and 'son' in rel_text:
                relationships["father_son"] += 1
            elif 'mother' in rel_text:
                if 'son' in rel_text:
                    relationships["mother_son"] += 1
                elif 'daughter' in rel_text:
                    relationships["mother_daughter"] += 1
            elif 'sister' in rel_text:
                if 'brother' not in rel_text:
                    relationships["sisters"] += 1
                else:
                    relationships["siblings"] += 1
            elif 'brother' in rel_text:
                relationships["brothers"] += 1

        # Calculate diagnostic ratios
        same_sex_rels = relationships["sisters"] + relationships["brothers"]
        cross_sex_rels = relationships["father_son"] + relationships["mother_son"] + relationships["siblings"]

        male_rels = relationships["father_son"] + relationships["brothers"]
        female_rels = relationships["mother_daughter"] + relationships["sisters"]

        return {
            "father_son_ratio": relationships["father_son"] / total_relationships if total_relationships > 0 else 0.0,
            "mother_daughter_ratio": relationships["mother_daughter"] / total_relationships if total_relationships > 0 else 0.0,
            "same_sex_ratio": same_sex_rels / total_relationships if total_relationships > 0 else 0.0,
            "cross_sex_ratio": cross_sex_rels / total_relationships if total_relationships > 0 else 0.0,
            "male_kinship_ratio": male_rels / total_relationships if total_relationships > 0 else 0.0,
            "female_kinship_ratio": female_rels / total_relationships if total_relationships > 0 else 0.0,
            "total_relationships": total_relationships
        }

    def calculate_sex_bias_metrics(self, site_name: str) -> Dict[str, float]:
        """
        Calculate sex bias metrics for inheritance pattern detection.

        Expected patterns:
        - Strongly patrilineal: Many related males, few related females
        - Strongly matrilineal: Many related females, few related males
        """
        site_data = self.processor.get_site_data(site_name)
        kinship_df = self.processor.get_site_kinship(site_name)

        # Basic sex ratios
        total_individuals = len(site_data)
        males = len(site_data[site_data['sex_clean'] == 'M'])
        females = len(site_data[site_data['sex_clean'] == 'F'])

        sex_ratio = males / females if females > 0 else float('inf')

        # Haplogroup sharing patterns
        y_sharing = self._calculate_haplogroup_sharing(site_data, 'y_chr_clean', 'M')
        mt_sharing = self._calculate_haplogroup_sharing(site_data, 'mt_dna_clean', None)

        # Related individuals by sex
        related_males, related_females = self._count_related_by_sex(kinship_df, site_data)

        return {
            "sex_ratio": sex_ratio,
            "male_proportion": males / total_individuals if total_individuals > 0 else 0.0,
            "female_proportion": females / total_individuals if total_individuals > 0 else 0.0,
            "y_haplogroup_sharing": y_sharing,
            "mt_haplogroup_sharing": mt_sharing,
            "related_male_proportion": related_males / males if males > 0 else 0.0,
            "related_female_proportion": related_females / females if females > 0 else 0.0,
        }

    def _calculate_haplogroup_sharing(self, site_data: pd.DataFrame, haplo_col: str, sex_filter: Optional[str]) -> float:
        """Calculate proportion of individuals sharing haplogroups."""
        if haplo_col not in site_data.columns:
            return 0.0

        data = site_data.copy()
        if sex_filter:
            data = data[data['sex_clean'] == sex_filter]

        valid_haplos = data[haplo_col].dropna()
        if len(valid_haplos) <= 1:
            return 0.0

        # Calculate sharing as 1 - normalized diversity
        diversity = self.calculate_nei_diversity(valid_haplos)
        max_diversity = (len(valid_haplos) - 1) / len(valid_haplos)

        return 1 - (diversity / max_diversity) if max_diversity > 0 else 0.0

    def _count_related_by_sex(self, kinship_df: pd.DataFrame, site_data: pd.DataFrame) -> Tuple[int, int]:
        """Count number of related males and females."""
        if kinship_df.empty:
            return 0, 0

        # Create mapping of individual IDs to sex
        id_to_sex = {}
        if self.processor.sample_id_col:
            for _, row in site_data.iterrows():
                sample_id = str(row[self.processor.sample_id_col])
                sex = row.get('sex_clean', 'Unknown')
                id_to_sex[sample_id] = sex

        related_individuals = set()
        for _, row in kinship_df.iterrows():
            related_individuals.add(str(row['Individual_1']))
            related_individuals.add(str(row['Individual_2']))

        related_males = sum(1 for ind in related_individuals if id_to_sex.get(ind) == 'M')
        related_females = sum(1 for ind in related_individuals if id_to_sex.get(ind) == 'F')

        return related_males, related_females

    def calculate_inheritance_signature(self, site_name: str, generations: int = 4) -> Dict[str, float]:
        """
        Calculate comprehensive inheritance pattern signature for a site.

        Returns dictionary with diagnostic statistics for model comparison.
        """
        site_data = self.processor.get_site_data(site_name)

        # Basic diversity measures
        y_diversity = 0.0
        mt_diversity = 0.0

        if 'y_chr_clean' in site_data.columns:
            y_valid = site_data[site_data['sex_clean'] == 'M']['y_chr_clean'].dropna()
            y_diversity = self.calculate_nei_diversity(y_valid)

        if 'mt_dna_clean' in site_data.columns:
            mt_valid = site_data['mt_dna_clean'].dropna()
            mt_diversity = self.calculate_nei_diversity(mt_valid)

        # Kinship ratios
        kinship_ratios = self.calculate_kinship_ratios(site_name)

        # Sex bias metrics
        sex_metrics = self.calculate_sex_bias_metrics(site_name)

        # Combine all measures into signature
        signature = {
            "site_name": site_name,
            "y_diversity": y_diversity,
            "mt_diversity": mt_diversity,
            "diversity_ratio": y_diversity / mt_diversity if mt_diversity > 0 else 0.0,
            **kinship_ratios,
            **sex_metrics,
        }

        return signature

    def classify_inheritance_pattern(self, signature: Dict[str, float]) -> Dict[str, float]:
        """
        Classify inheritance pattern based on signature statistics.

        Returns probabilities for each inheritance system.
        """
        # Define diagnostic thresholds based on theoretical expectations
        scores = {
            "strongly_patrilineal": 0.0,
            "weakly_patrilineal": 0.0,
            "balanced": 0.0,
            "weakly_matrilineal": 0.0,
            "strongly_matrilineal": 0.0
        }

        # Y-chromosome diversity scoring
        y_div = signature["y_diversity"]
        if y_div < 0.1:  # Very low Y diversity
            scores["strongly_patrilineal"] += 2.0
            scores["weakly_patrilineal"] += 1.0
        elif y_div > 0.8:  # High Y diversity
            scores["strongly_matrilineal"] += 2.0
            scores["weakly_matrilineal"] += 1.0
        else:  # Moderate Y diversity
            scores["balanced"] += 1.0
            scores["weakly_patrilineal"] += 0.5
            scores["weakly_matrilineal"] += 0.5

        # mtDNA diversity scoring (inverse relationship to Y for patrilineal)
        mt_div = signature["mt_diversity"]
        if mt_div > 0.8:  # High mt diversity
            scores["strongly_patrilineal"] += 1.0
            scores["weakly_patrilineal"] += 0.5
        elif mt_div < 0.3:  # Low mt diversity
            scores["strongly_matrilineal"] += 1.0
            scores["weakly_matrilineal"] += 0.5

        # Kinship pattern scoring
        father_son_ratio = signature["father_son_ratio"]
        mother_daughter_ratio = signature["mother_daughter_ratio"]

        if father_son_ratio > 0.3:
            scores["strongly_patrilineal"] += 1.5
            scores["weakly_patrilineal"] += 1.0

        if mother_daughter_ratio > 0.3:
            scores["strongly_matrilineal"] += 1.5
            scores["weakly_matrilineal"] += 1.0

        # Sex bias scoring
        male_kinship = signature["male_kinship_ratio"]
        female_kinship = signature["female_kinship_ratio"]

        if male_kinship > female_kinship * 2:
            scores["strongly_patrilineal"] += 1.0
            scores["weakly_patrilineal"] += 0.5
        elif female_kinship > male_kinship * 2:
            scores["strongly_matrilineal"] += 1.0
            scores["weakly_matrilineal"] += 0.5
        else:
            scores["balanced"] += 1.0

        # Normalize to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {k: v / total_score for k, v in scores.items()}
        else:
            # Default to uniform distribution if no clear signal
            probabilities = {k: 0.2 for k in scores.keys()}

        return probabilities

    def analyze_all_sites(self) -> pd.DataFrame:
        """Analyze all sites and return comprehensive results."""
        sites = self.processor.get_sites()
        results = []

        for site in sites:
            try:
                signature = self.calculate_inheritance_signature(site)
                probabilities = self.classify_inheritance_pattern(signature)

                result = {
                    **signature,
                    **{f"prob_{k}": v for k, v in probabilities.items()}
                }
                results.append(result)

            except Exception as e:
                print(f"Error analyzing site {site}: {e}")
                continue

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Test the inheritance pattern analysis
    from data_preprocessing import load_and_preprocess_data

    processor = load_and_preprocess_data()
    analyzer = InheritancePatternAnalyzer(processor)

    print("INHERITANCE PATTERN ANALYSIS")
    print("=" * 60)

    # Analyze all sites
    results_df = analyzer.analyze_all_sites()

    # Display results
    for _, row in results_df.iterrows():
        print(f"\nSite: {row['site_name']}")
        print(f"Y diversity: {row['y_diversity']:.3f}, mtDNA diversity: {row['mt_diversity']:.3f}")
        print(f"Kinship - Father-son: {row['father_son_ratio']:.2f}, Mother-daughter: {row['mother_daughter_ratio']:.2f}")

        # Show top inheritance pattern
        prob_cols = [col for col in row.index if col.startswith('prob_')]
        probs = {col.replace('prob_', ''): row[col] for col in prob_cols}
        best_pattern = max(probs, key=probs.get)
        print(f"Most likely pattern: {best_pattern} (p={probs[best_pattern]:.2f})")

    # Save detailed results
    results_df.to_csv('inheritance_analysis_results.csv', index=False)
    print(f"\nDetailed results saved to 'inheritance_analysis_results.csv'")