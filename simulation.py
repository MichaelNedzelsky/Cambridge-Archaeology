# RomanoBritishInheritance Agent-Based Model
# A simulation to test hypotheses of social structure in Roman-era Britain.

import random
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
# These parameters can be adjusted to run different scenarios.
SIMULATION_CONFIG = {
    'residence_rule': 'mixed',  # 'patrilocal', 'matrilocal', or 'mixed'
    'end_year': 400,
    'p_birth': 0.08,  # Annual probability of birth for a married female
    'p_burial_inheritor': 0.8,
    'p_burial_non_inheritor': 0.2,
    'p_dna_success': 0.46,
    'settlements': [],
    # Simplified haplogroup frequencies for Iron Age Britain
    'y_haplogroups': {'R1b': 0.7, 'I2': 0.2, 'I1': 0.05, 'G2a': 0.05},
    'mt_haplogroups': {'H': 0.4, 'U': 0.2, 'J': 0.1, 'K': 0.1, 'T': 0.1, 'Other': 0.1}
}

class Agent:
    """Represents an individual in the simulation."""
    def __init__(self, agent_id, sex, age, settlement, mother=None, father=None):
        self.id = agent_id
        self.sex = sex
        self.age = age
        self.home_settlement = settlement
        self.is_alive = True
        self.death_year = -1
        
        # Kinship
        self.mother = mother
        self.father = father
        self.spouse = None
        self.children = []
        
        # Social Status
        self.is_inheritor = False
        
        # Genetics
        if self.sex == 'M':
            self.y_haplogroup = father.y_haplogroup if father else random.choices(list(SIMULATION_CONFIG['y_haplogroups'].keys()), weights=list(SIMULATION_CONFIG['y_haplogroups'].values()))
        else:
            self.y_haplogroup = None
        self.mt_haplogroup = mother.mt_haplogroup if mother else random.choices(list(SIMULATION_CONFIG['mt_haplogroups'].keys()), weights=list(SIMULATION_CONFIG['mt_haplogroups'].values()))

    def age_one_year(self):
        if self.is_alive:
            self.age += 1

    def check_mortality(self, current_year):
        """Determines if an agent dies based on age-structured probability."""
        if not self.is_alive:
            return
        
        # Simplified mortality curve
        if self.age < 5: p_death = 0.05
        elif self.age > 60: p_death = 0.2
        else: p_death = 0.02
        
        if random.random() < p_death:
            self.is_alive = False
            self.death_year = current_year
            if self.spouse:
                self.spouse.spouse = None
            self.home_settlement.remove_agent(self)

class Settlement:
    """Represents a settlement location."""
    def __init__(self, name, max_pop, status, start_year, end_year):
        self.name = name
        self.max_pop = max_pop
        self.status = status
        self.start_year = start_year
        self.end_year = end_year
        self.agents = []

    @property
    def current_population(self):
        return len(self.agents)

    def add_agent(self, agent):
        if self.current_population < self.max_pop:
            self.agents.append(agent)
            agent.home_settlement = self
            return True
        return False

    def remove_agent(self, agent):
        if agent in self.agents:
            self.agents.remove(agent)

class Simulation:
    """Main controller for the simulation."""
    def __init__(self, config):
        self.config = config
        self.year = 0
        self.next_agent_id = 0
        self.settlements = {s['name']: Settlement(s['name'], s['max_pop'], s['status'], s['start_year'], s['end_year']) for s in config['settlements']}
        self.all_agents = []
        self.dead_agents = []
        self._initialize_population()

    def _get_new_agent_id(self):
        self.next_agent_id += 1
        return self.next_agent_id

    def _initialize_population(self):
        """Creates the starting population for each settlement."""
        for sett_config in self.config['settlements']:
            settlement = self.settlements[sett_config['name']]
            if self.year >= settlement.start_year:
                for _ in range(sett_config['start_pop']):
                    sex = random.choice(['M', 'F'])
                    age = random.randint(1, 40)
                    agent = Agent(self._get_new_agent_id(), sex, age, settlement)
                    settlement.add_agent(agent)
                    self.all_agents.append(agent)

    def run_step(self):
        """Runs one year of the simulation."""
        self.year += 1
        
        # 1. Age agents and check for mortality
        for agent in list(self.all_agents):
            if agent.is_alive:
                agent.age_one_year()
                agent.check_mortality(self.year)
        
        self.dead_agents.extend([a for a in self.all_agents if not a.is_alive and a not in self.dead_agents])
        
        # 2. Marriage
        self._process_marriages()
        
        # 3. Reproduction
        self._process_reproduction()
        
        # 4. Inheritance
        self._process_inheritance()

    def _process_marriages(self):
        """Handles marriage and residence changes based on the rule."""
        unmarried_females = [a for a in self.all_agents if a.is_alive and a.sex == 'F' and a.age >= 16 and not a.spouse]
        random.shuffle(unmarried_females)

        for female in unmarried_females:
            potential_partners = [p for p in self.all_agents if p.is_alive and p.sex == 'M' and p.age >= 16 and not p.spouse and abs(p.age - female.age) < 10]
            if not potential_partners:
                continue
            
            partner = random.choice(potential_partners)
            
            # Form marriage bond
            female.spouse = partner
            partner.spouse = female
            
            # Apply residence rule
            rule = self.config['residence_rule']
            if rule == 'mixed':
                rule = 'patrilocal' if female.home_settlement.status == 'high' else 'matrilocal'
            
            if female.home_settlement != partner.home_settlement:
                if rule == 'patrilocal':
                    # Female moves
                    female.home_settlement.remove_agent(female)
                    partner.home_settlement.add_agent(female)
                elif rule == 'matrilocal':
                    # Male moves
                    partner.home_settlement.remove_agent(partner)
                    female.home_settlement.add_agent(partner)

    def _process_reproduction(self):
        """Handles births."""
        fertile_females = [a for a in self.all_agents if a.is_alive and a.sex == 'F' and 18 <= a.age <= 40 and a.spouse]
        for female in fertile_females:
            if random.random() < self.config['p_birth']:
                sex = random.choice(['M', 'F'])
                child = Agent(self._get_new_agent_id(), sex, 0, female.home_settlement, mother=female, father=female.spouse)
                if female.home_settlement.add_agent(child):
                    self.all_agents.append(child)
                    female.children.append(child)
                    female.spouse.children.append(child)

    def _process_inheritance(self):
        """Assigns inheritor status based on the residence rule."""
        for agent in self.all_agents:
            agent.is_inheritor = False # Reset each year
            
        parent_pairs = set([(a.mother, a.father) for a in self.all_agents if a.mother and a.father])
        
        for mother, father in parent_pairs:
            if not mother.is_alive or not father.is_alive:
                continue

            rule = self.config['residence_rule']
            if rule == 'mixed':
                rule = 'patrilocal' if mother.home_settlement.status == 'high' else 'matrilocal'

            if rule == 'patrilocal':
                sons = [c for c in mother.children if c.sex == 'M' and c.is_alive]
                if sons:
                    inheritor = min(sons, key=lambda x: x.age) # Eldest son inherits
                    inheritor.is_inheritor = True
            elif rule == 'matrilocal':
                daughters = [c for c in mother.children if c.sex == 'F' and c.is_alive]
                if daughters:
                    inheritor = min(daughters, key=lambda x: x.age) # Eldest daughter inherits
                    inheritor.is_inheritor = True

    def run_simulation(self):
        """Runs the simulation from year 0 to the end year."""
        while self.year < self.config['end_year']:
            self.run_step()
        print(f"Simulation finished at year {self.year}.")
        return self.generate_cemetery()

    def generate_cemetery(self):
        """Generates the final 'observed' cemetery data."""
        cemetery_population = []
        for agent in self.dead_agents:
            settlement_end_year = self.settlements[agent.home_settlement.name].end_year
            # Only include agents who died during the settlement's occupation
            if agent.death_year <= settlement_end_year:
                p_burial = self.config['p_burial_inheritor'] if agent.is_inheritor else self.config['p_burial_non_inheritor']
                if random.random() < p_burial:
                    cemetery_population.append(agent)
        
        # Apply DNA success filter
        num_to_sample = int(len(cemetery_population) * self.config['p_dna_success'])
        observed_skeletons = random.sample(cemetery_population, k=min(num_to_sample, len(cemetery_population)))
        
        # Convert to DataFrame for analysis
        data = []
        for agent in observed_skeletons:
            data.append({
                'id': agent.id,
                'settlement': agent.home_settlement.name,
                'sex': agent.sex,
                'age_at_death': agent.age,
                'y_haplogroup': agent.y_haplogroup,
                'mt_haplogroup': agent.mt_haplogroup,
                'mother_id': agent.mother.id if agent.mother else None,
                'father_id': agent.father.id if agent.father else None
            })
        return pd.DataFrame(data)

# --- How to run the simulation ---
# sim = Simulation(SIMULATION_CONFIG)
# results_df = sim.run_simulation()
# print(results_df.head())
# print(f"\nTotal skeletons in virtual cemetery: {len(results_df)}")
