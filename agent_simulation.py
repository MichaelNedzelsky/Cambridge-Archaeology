"""
Agent-based simulation for testing inheritance systems in archaeological populations.

Simulates virtual cemeteries under different inheritance patterns:
- Strongly matrilineal
- Weakly matrilineal
- Balanced
- Weakly patrilineal
- Strongly patrilineal

Based on theoretical framework from description.md and archaeological parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import random
from inheritance_statistics import InheritancePatternAnalyzer
from data_preprocessing import SiteDataProcessor
from site_parameters import get_site_population, DEFAULT_POPULATION_SIZE
from global_parameters import (
    BURIAL_PROBABILITY, ADNA_SUCCESS_RATE, DEFAULT_GENERATIONS,
    DEFAULT_Y_HAPLOGROUPS, DEFAULT_MT_HAPLOGROUPS,
    get_inheritance_probabilities
)


@dataclass
class Individual:
    """Represents an individual in the simulation."""
    id: str
    sex: str  # 'M' or 'F'
    generation: int
    age: int
    is_inheritor: bool
    y_haplogroup: Optional[str] = None
    mt_haplogroup: Optional[str] = None
    father_id: Optional[str] = None
    mother_id: Optional[str] = None
    spouse_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    is_buried: bool = False
    burial_site: Optional[str] = None


@dataclass
class SimulationParameters:
    """Parameters for inheritance system simulation."""
    inheritance_system: str
    generations: int = field(default=DEFAULT_GENERATIONS)
    population_per_generation: int = 20
    site_name: Optional[str] = None  # Site name for site-specific population
    burial_probability: float = field(default=BURIAL_PROBABILITY)
    adna_success_rate: float = field(default=ADNA_SUCCESS_RATE)
    inheritance_probability_male: float = 0.5
    inheritance_probability_female: float = 0.5
    starting_haplogroups_y: List[str] = field(default_factory=lambda: DEFAULT_Y_HAPLOGROUPS.copy())
    starting_haplogroups_mt: List[str] = field(default_factory=lambda: DEFAULT_MT_HAPLOGROUPS.copy())


class InheritanceSimulator:
    """Simulates populations under different inheritance systems."""

    def __init__(self, params: SimulationParameters):
        self.params = params
        self.individuals: Dict[str, Individual] = {}
        self.next_id = 1
        self.haplogroup_pool_y = params.starting_haplogroups_y.copy()
        self.haplogroup_pool_mt = params.starting_haplogroups_mt.copy()

        # Set site-specific population if site name provided
        if params.site_name:
            try:
                self.params.population_per_generation = get_site_population(params.site_name)
            except ValueError:
                # Fall back to default if site not found
                self.params.population_per_generation = DEFAULT_POPULATION_SIZE

        # Set inheritance probabilities based on system
        self._configure_inheritance_system()

    def _configure_inheritance_system(self):
        """Configure inheritance probabilities based on system type."""
        system = self.params.inheritance_system

        try:
            male_prob, female_prob = get_inheritance_probabilities(system)
            self.params.inheritance_probability_male = male_prob
            self.params.inheritance_probability_female = female_prob
        except ValueError as e:
            # Re-raise with more context
            raise ValueError(f"Failed to configure inheritance system: {e}")

    def _generate_id(self) -> str:
        """Generate unique individual ID."""
        id_str = f"IND_{self.next_id:04d}"
        self.next_id += 1
        return id_str

    def _assign_haplogroup(self, individual: Individual, father: Optional[Individual], mother: Optional[Individual]):
        """Assign haplogroups based on inheritance rules (no migration)."""
        # Y-chromosome (males only, from father)
        if individual.sex == 'M':
            if father and father.y_haplogroup:
                individual.y_haplogroup = father.y_haplogroup
            else:
                # New haplogroup from founding population
                individual.y_haplogroup = random.choice(self.haplogroup_pool_y)

        # Mitochondrial DNA (from mother)
        if mother and mother.mt_haplogroup:
            individual.mt_haplogroup = mother.mt_haplogroup
        else:
            # New haplogroup from founding population
            individual.mt_haplogroup = random.choice(self.haplogroup_pool_mt)

    def _determine_inheritance(self, individual: Individual) -> bool:
        """Determine if individual inherits based on sex and system."""
        if individual.sex == 'M':
            return random.random() < self.params.inheritance_probability_male
        else:
            return random.random() < self.params.inheritance_probability_female

    def _simulate_burial(self, individual: Individual) -> bool:
        """Determine if individual is buried at the site."""
        base_prob = self.params.burial_probability

        # Modify probability based on inheritance status
        if individual.is_inheritor:
            base_prob *= 1.5  # Inheritors more likely to be buried locally

        # Age-based adjustment (adults more likely to be buried)
        if individual.age >= 16:
            base_prob *= 1.2

        return random.random() < min(base_prob, 1.0)

    def _simulate_adna_success(self) -> bool:
        """Determine if aDNA extraction succeeds."""
        return random.random() < self.params.adna_success_rate

    def create_founding_generation(self) -> List[Individual]:
        """Create the founding generation with diverse haplogroups."""
        founders = []

        for i in range(self.params.population_per_generation):
            sex = 'M' if i < self.params.population_per_generation // 2 else 'F'
            individual = Individual(
                id=self._generate_id(),
                sex=sex,
                generation=0,
                age=25 + random.randint(0, 20),  # Adults aged 25-45
                is_inheritor=True  # Founders are all inheritors
            )

            # Assign diverse founding haplogroups
            if sex == 'M':
                individual.y_haplogroup = random.choice(self.haplogroup_pool_y)
            individual.mt_haplogroup = random.choice(self.haplogroup_pool_mt)

            self.individuals[individual.id] = individual
            founders.append(individual)

        return founders

    def create_offspring(self, father: Individual, mother: Individual, generation: int) -> List[Individual]:
        """Create offspring from a mated pair."""
        num_children = random.choices([0, 1, 2, 3, 4], weights=[0.1, 0.2, 0.4, 0.2, 0.1])[0]
        children = []

        for _ in range(num_children):
            sex = random.choice(['M', 'F'])
            child = Individual(
                id=self._generate_id(),
                sex=sex,
                generation=generation,
                age=random.randint(0, 20),  # Children and young adults
                is_inheritor=False,  # Will be determined later
                father_id=father.id,
                mother_id=mother.id
            )

            # Assign haplogroups
            self._assign_haplogroup(child, father, mother)

            # Determine inheritance status
            child.is_inheritor = self._determine_inheritance(child)

            # Update parent records
            father.children.append(child.id)
            mother.children.append(child.id)

            self.individuals[child.id] = child
            children.append(child)

        return children

    def simulate_mating(self, generation_individuals: List[Individual]) -> List[Individual]:
        """Simulate mating within a generation."""
        males = [ind for ind in generation_individuals if ind.sex == 'M' and ind.age >= 16]
        females = [ind for ind in generation_individuals if ind.sex == 'F' and ind.age >= 16]

        # Randomly pair individuals (simplified mating model)
        random.shuffle(males)
        random.shuffle(females)

        offspring = []
        min_pairs = min(len(males), len(females))

        for i in range(min_pairs):
            if random.random() < 0.8:  # 80% chance of mating
                children = self.create_offspring(males[i], females[i], generation_individuals[0].generation + 1)
                offspring.extend(children)

                # Record spouse relationships
                males[i].spouse_id = females[i].id
                females[i].spouse_id = males[i].id

        return offspring

    def run_simulation(self) -> Dict:
        """Run complete simulation and return results."""
        # Create founding generation
        current_generation = self.create_founding_generation()
        all_individuals = current_generation.copy()

        # Simulate subsequent generations
        for gen in range(1, self.params.generations + 1):
            offspring = self.simulate_mating(current_generation)
            all_individuals.extend(offspring)
            current_generation = offspring

        # Determine burials and aDNA success
        buried_individuals = []
        adna_successful = []

        for individual in all_individuals:
            if self._simulate_burial(individual):
                individual.is_buried = True
                individual.burial_site = "SIMULATED_SITE"
                buried_individuals.append(individual)

                if self._simulate_adna_success():
                    adna_successful.append(individual)

        # Generate results
        results = self._generate_simulation_results(all_individuals, buried_individuals, adna_successful)

        return results

    def _generate_simulation_results(self, all_individuals: List[Individual],
                                   buried: List[Individual], adna_successful: List[Individual]) -> Dict:
        """Generate comprehensive simulation results."""

        # Convert to DataFrame for analysis
        buried_data = []
        for ind in buried:
            has_adna = ind in adna_successful
            buried_data.append({
                'Individual_ID': ind.id,
                'Sex': ind.sex,
                'Generation': ind.generation,
                'Age': ind.age,
                'Is_Inheritor': ind.is_inheritor,
                'Y_Haplogroup': ind.y_haplogroup if has_adna else np.nan,
                'mt_Haplogroup': ind.mt_haplogroup if has_adna else np.nan,
                'Father_ID': ind.father_id,
                'Mother_ID': ind.mother_id,
                'Has_aDNA': has_adna
            })

        df = pd.DataFrame(buried_data)

        # Calculate statistics
        stats = self._calculate_simulation_statistics(df, all_individuals)

        # Calculate kinship relationships
        kinship_pairs = self._calculate_kinship_relationships(df, all_individuals)

        results = {
            'parameters': self.params,
            'buried_individuals': df,
            'statistics': stats,
            'kinship_pairs': kinship_pairs,
            'total_population': len(all_individuals),
            'buried_count': len(buried),
            'adna_count': len(adna_successful),
            'success': True
        }

        return results

    def _calculate_simulation_statistics(self, df: pd.DataFrame, all_individuals: List[Individual]) -> Dict:
        """Calculate key statistics from simulation results."""
        stats = {}

        # Basic counts
        stats['total_buried'] = len(df)
        stats['males'] = len(df[df['Sex'] == 'M'])
        stats['females'] = len(df[df['Sex'] == 'F'])

        # aDNA statistics
        adna_df = df[df['Has_aDNA']]

        # Y-chromosome diversity
        y_haplos = adna_df[adna_df['Sex'] == 'M']['Y_Haplogroup'].dropna()
        stats['y_diversity'] = self._calculate_nei_diversity(y_haplos)
        stats['y_samples'] = len(y_haplos)
        stats['y_unique'] = len(y_haplos.unique()) if len(y_haplos) > 0 else 0

        # mtDNA diversity
        mt_haplos = adna_df['mt_Haplogroup'].dropna()
        stats['mt_diversity'] = self._calculate_nei_diversity(mt_haplos)
        stats['mt_samples'] = len(mt_haplos)
        stats['mt_unique'] = len(mt_haplos.unique()) if len(mt_haplos) > 0 else 0

        # Inheritance patterns
        inheritors = df[df['Is_Inheritor']]
        stats['inheritor_male_ratio'] = len(inheritors[inheritors['Sex'] == 'M']) / len(inheritors) if len(inheritors) > 0 else 0
        stats['inheritor_female_ratio'] = len(inheritors[inheritors['Sex'] == 'F']) / len(inheritors) if len(inheritors) > 0 else 0

        return stats

    def _calculate_nei_diversity(self, haplogroups: pd.Series) -> float:
        """Calculate Nei's diversity index."""
        if len(haplogroups) <= 1:
            return 0.0

        counts = haplogroups.value_counts()
        n = len(haplogroups)
        freq_sum_sq = sum((count / n) ** 2 for count in counts)

        return (n / (n - 1)) * (1 - freq_sum_sq)

    def _calculate_kinship_relationships(self, df: pd.DataFrame, all_individuals: List[Individual]) -> List[Dict]:
        """Calculate kinship relationships between buried individuals."""
        individuals_dict = {ind.id: ind for ind in all_individuals}
        kinship_pairs = []

        buried_ids = set(df['Individual_ID'])

        for i, ind1_id in enumerate(buried_ids):
            for ind2_id in list(buried_ids)[i+1:]:
                relationship = self._determine_relationship(ind1_id, ind2_id, individuals_dict)

                if relationship:
                    ind1 = individuals_dict[ind1_id]
                    ind2 = individuals_dict[ind2_id]

                    kinship_pairs.append({
                        'Individual_1': ind1_id,
                        'Individual_2': ind2_id,
                        'Relationship': relationship['type'],
                        'Degree': relationship['degree'],
                        'Sex_1': ind1.sex,
                        'Sex_2': ind2.sex,
                        'Y_Match': (ind1.y_haplogroup == ind2.y_haplogroup) if ind1.y_haplogroup and ind2.y_haplogroup else False,
                        'mt_Match': (ind1.mt_haplogroup == ind2.mt_haplogroup) if ind1.mt_haplogroup and ind2.mt_haplogroup else False
                    })

        return kinship_pairs

    def _determine_relationship(self, id1: str, id2: str, individuals_dict: Dict[str, Individual]) -> Optional[Dict]:
        """Determine relationship between two individuals."""
        ind1 = individuals_dict[id1]
        ind2 = individuals_dict[id2]

        # Parent-offspring (1st degree)
        if ind1.father_id == id2 or ind1.mother_id == id2:
            return {'type': 'Parent-Offspring', 'degree': 1}
        if ind2.father_id == id1 or ind2.mother_id == id1:
            return {'type': 'Parent-Offspring', 'degree': 1}

        # Siblings (1st degree)
        if (ind1.father_id and ind1.father_id == ind2.father_id) or (ind1.mother_id and ind1.mother_id == ind2.mother_id):
            if ind1.father_id == ind2.father_id and ind1.mother_id == ind2.mother_id:
                return {'type': 'Full Siblings', 'degree': 1}
            else:
                return {'type': 'Half Siblings', 'degree': 2}

        # Grandparent-grandchild (2nd degree)
        # Check if one is grandparent of the other
        for ind, other in [(ind1, ind2), (ind2, ind1)]:
            if other.father_id in individuals_dict:
                grandparent_f = individuals_dict[other.father_id]
                if grandparent_f.father_id == ind.id or grandparent_f.mother_id == ind.id:
                    return {'type': 'Grandparent-Grandchild', 'degree': 2}
            if other.mother_id in individuals_dict:
                grandparent_m = individuals_dict[other.mother_id]
                if grandparent_m.father_id == ind.id or grandparent_m.mother_id == ind.id:
                    return {'type': 'Grandparent-Grandchild', 'degree': 2}

        # For simplicity, not implementing more distant relationships
        return None


class BatchSimulation:
    """Run batch simulations for model comparison."""

    def __init__(self, site_data: Dict, n_simulations: int = 100):
        self.site_data = site_data
        self.n_simulations = n_simulations

    def run_batch_for_all_systems(self, site_name: str) -> Dict:
        """Run batch simulations for all inheritance systems."""
        systems = ['strongly_patrilineal', 'weakly_patrilineal', 'balanced',
                  'weakly_matrilineal', 'strongly_matrilineal']

        results = {}

        for system in systems:
            print(f"Running {self.n_simulations} simulations for {system} system...")
            system_results = []

            for i in range(self.n_simulations):
                params = SimulationParameters(
                    inheritance_system=system,
                    generations=4,
                    population_per_generation=20,
                    burial_probability=0.8,
                    adna_success_rate=0.7
                )

                try:
                    simulator = InheritanceSimulator(params)
                    result = simulator.run_simulation()
                    system_results.append(result)
                except Exception as e:
                    print(f"Simulation failed for {system}, run {i}: {e}")
                    continue

            results[system] = system_results
            print(f"Completed {len(system_results)} successful simulations for {system}")

        return results


if __name__ == "__main__":
    # Test single simulation
    print("Testing single simulation...")

    params = SimulationParameters(
        inheritance_system="strongly_patrilineal",
        generations=3,
        population_per_generation=15
    )

    simulator = InheritanceSimulator(params)
    result = simulator.run_simulation()

    print(f"Simulation completed successfully!")
    print(f"Total population: {result['total_population']}")
    print(f"Buried individuals: {result['buried_count']}")
    print(f"aDNA successful: {result['adna_count']}")
    print(f"Kinship pairs found: {len(result['kinship_pairs'])}")
    print(f"Y-chromosome diversity: {result['statistics']['y_diversity']:.3f}")
    print(f"mtDNA diversity: {result['statistics']['mt_diversity']:.3f}")

    # Show some kinship relationships
    if result['kinship_pairs']:
        print("\nKinship relationships found:")
        for pair in result['kinship_pairs'][:5]:  # Show first 5
            print(f"  {pair['Individual_1']} - {pair['Individual_2']}: {pair['Relationship']}")

    print("\nSingle simulation test completed successfully!")