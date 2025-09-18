# Cambridge-Archaeology
A collection of scripts 

```
python analyze_adna.py --input combined_grouped.csv --kinship "Cambridshire aDNA summary data.xlsx - DNA kinship details.csv" --output final_summary.csv
```
| Site                | Kinship Pairs (Degree)                                                                  | Y-chr Samples (N) | Y-chr Diversity (H) | mtDNA Samples (N) | mtDNA Diversity (H) |
|---------------------|-----------------------------------------------------------------------------------------|------------------:|--------------------:|------------------:|--------------------:|
| Arbury              | 1st                                                                                     | 3                | 1.00               | 6                | 1.00               |
| Duxford             | Siblings (1st, x2, 2nd), Cousins (2nd, 3rd), Avuncular (3rd), Grandparent Grandchild (3rd), 1st, x2, 4th | 9                | 1.00               | 20               | 0.98               |
| Fenstanton          | Siblings (1st), 2nd                                                                      | 7                | 0.86               | 20               | 0.92               |
| Knobbs              | 2nd                                                                                     | 3                | 1.00               | 20               | 0.76               |
| Northwest Cambridge | Grandparent Grandchild (2nd), 4th, 2nd                                                   | 4                | 1.00               | 8                | 1.00               |
| Vicar's Farm        | Siblings (1st, x2, 2nd, x2), Cousins (2nd, x2)                                           | 4                | 1.00               | 17               | 0.96               |


## A Generative Model of Romano-British Farmstead Demography and Genetics

### Conceptual Framework
The proposed model, RomanoBritishInheritance, is an Agent-Based Model designed to simulate the key demographic, genetic, and social processes that shaped the Cambridgeshire cemeteries. The model's objective is to generate a "virtual archaeological record"—a simulated cemetery population with associated genetic data—that can be statistically compared against the empirical benchmark. This generative approach provides a principled way to test which underlying social rules are most likely to produce the observed patterns of kinship and genetic diversity.

### Model Components
The model consists of two primary components: the environment and the agents that inhabit it.

#### The Environment
The environment is composed of discrete Settlement objects, each parameterized according to the archaeological context provided for the Cambridgeshire sites.

- **name:** A string identifier (e.g., 'Duxford', 'Arbury').
- **max_population:** An integer representing the estimated maximum carrying capacity of the farmstead (e.g., 30 for Duxford, 50 for Vicar's Farm).
- **timeline:** A tuple defining the start and end year of the settlement's occupation.
- **status:** A categorical variable ('low', 'medium', 'high') to allow for status-dependent rules.

#### The Agents
Each Agent in the simulation represents an individual human with a set of attributes that evolve over their lifetime.

- **id, age, sex:** Basic demographic attributes.
- **y_haplogroup, mt_haplogroup:** Genetic markers inherited from parents.
- **is_alive, is_married:** Boolean flags tracking life status.
- **kin_links:** A dictionary of pointers to the agent's mother, father, spouse, and children, forming the basis of the kin network.
- **social_status:** A key attribute that determines their social and economic role (e.g., 'inheritor', 'non-inheritor', 'immigrant').
- **home_settlement:** A pointer to the Settlement object where the agent currently resides.

### Processes and Rules (The Model's Engine)
The simulation proceeds in discrete annual time-steps, during which a series of demographic and social processes are executed for each agent.

#### Demographics
Agents age by one year at each time-step. Mortality is governed by an age-structured probability distribution, reflecting higher mortality rates in infancy and old age. Reproduction is a probabilistic event for married female agents within a defined reproductive age range (e.g., 16-45 years).

#### Genetic Inheritance
Upon birth, an agent's genetic markers are inherited from its parents. A male agent receives his y_haplogroup from his father and mt_haplogroup from his mother. A female agent receives her mt_haplogroup from her mother. The initial population of agents at the start of the simulation is seeded with haplogroups drawn from a frequency distribution representative of the preceding Iron Age population to provide a realistic genetic starting point.

#### Marriage and Residence
This module is the core of the hypothesis-testing framework. The simulation can be configured to run under one of several residence_rule scenarios:

- **Patrilocal:** A female agent of marriageable age seeks a male partner, prioritizing males within her own settlement. If no suitable partner is found locally, she searches in neighboring settlements. Upon marriage, she moves to her husband's home_settlement.
- **Matrilocal:** A female agent of marriageable age seeks a male partner. Upon marriage to a male from another settlement, he moves to her home_settlement.
- **Bilocal/Mixed:** The residence rule is determined probabilistically or is conditioned on agent/settlement attributes. For instance, in a "Romanization" scenario, marriages involving agents from high-status settlements follow a patrilocal rule, while those in low-status settlements follow a matrilocal rule.

#### Inheritance and Burial
The model explicitly implements the "inheritor burial bias" hypothesis, which posits that individuals who inherit land rights are more likely to be buried in a formal cemetery on that land. This creates the critical filter between the total living population of the simulation and the final, observable cemetery sample.

An inheritance_rule is tied to the residence_rule: in a patrilocal simulation, one son per family is designated the inheritor; in a matrilocal simulation, one daughter is. At the end of the simulation's timeline, the model generates the final cemetery. Agents who died during the occupation period are sampled for inclusion based on their status. Inheritors have a high probability of burial (p_burial_inheritor), while non-inheriting family members and immigrants have a significantly lower probability (p_burial_non_inheritor). This mechanism directly simulates the social process thought to be responsible for the formation of the archaeological record.

#### Data Degradation and Observation
To ensure a fair comparison between simulated and real data, the model simulates the process of aDNA degradation and incomplete recovery. After the "true" cemetery population is generated, it is passed through an observation filter. This filter first subsamples the buried population to mimic the observed success rate of aDNA testing (approx. 46%). For this subsample, a degradation_function is applied, which probabilistically reduces the specificity of haplogroup assignments (e.g., a "true" haplogroup of 'R1b-P312' might be observed as 'R1b-M269' or simply 'R'). This ensures that the simulated data exhibits the same types of noise and ambiguity as the real archaeological dataset, making the subsequent statistical comparison valid.

The code is organized into three main classes: Agent, Settlement, and Simulation. This modular structure allows for clear separation of concerns and facilitates future extensions.

## Class Definitions
Agent Class: This class stores all individual-level attributes (ID, age, sex, haplogroups, kin links, etc.) and contains methods for individual actions like age_one_year(), find_partner(), reproduce(), and check_mortality().

Settlement Class: This class manages the list of agents residing within it. It contains methods to add_agent(), remove_agent(), and enforce the max_population constraint.

Simulation Class: This is the main controller. It initializes the simulation world with settlements and a starting population, iterates through time-steps (run_simulation()), applies the global social rules for marriage and inheritance, and, at the end, generates the final "observed" cemetery data (generate_cemetery()) and calculates the required summary statistics (calculate_summary_stats()).

## Simulating Scenarios: Testing Hypotheses of Social Organization

### Experimental Design
To test the central research question, this study employs a structured experimental design comprising three distinct scenarios. Each scenario represents a competing hypothesis about the social organization of Roman-era Cambridgeshire. To account for the inherent stochasticity of the model, each scenario will be run multiple times (an ensemble of runs), and the distribution of outcomes will be analyzed.

**Scenario 1: Uniform Patrilocality.** This scenario models a complete cultural shift to Roman norms. All six simulated settlements operate under a strict patrilocal residence and patrilineal inheritance rule throughout their occupation.

**Scenario 2: Uniform Matrilocality.** This scenario models the persistence of hypothesized prehistoric traditions. All six settlements operate under a strict matrilocal residence and matrilineal inheritance rule.

**Scenario 3: Mixed System (Status-Based).** This scenario tests the "Romanization" hypothesis, where cultural adoption is heterogeneous. High-status settlements (Arbury, Vicar's Farm) are configured to be patrilocal, reflecting an adoption of elite Roman customs. Lower-status and chronologically earlier settlements (Duxford, Knobbs Farm, etc.) are configured to be matrilocal, reflecting the persistence of older traditions.

### Summary Statistics for Comparison
For each simulated cemetery generated by the model, a vector of summary statistics is calculated. These statistics are chosen because they are known to produce distinct signatures under different residence patterns, providing a quantitative basis for comparing the model output to the empirical data.

References:
- [The Genetic Signature of Sex-Biased Migration in Patrilocal and Matrilocal Populations](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000973)
- [Human mtDNA and Y-chromosome Variation Is Correlated with Matrilocal versus Patrilocal Residence](https://www.researchgate.net/publication/11818392_Human_mtDNA_and_Y-chromosome_Variation_Is_Correlated_with_Matrilocal_versus_Patrilocal_Residence)
- [Oota et al. (2001) - Human mtDNA and Y-chromosome variation](https://os.pennds.org/archaeobib_filestore/pdf_articles/NatGenet/2001_29_1_Ootaetal.pdf)
- [Reduced Y-Chromosome, but Not Mitochondrial DNA, Diversity in Human Populations from West New Guinea](https://pmc.ncbi.nlm.nih.gov/articles/PMC379223/)

**Stat 1: Y-chromosome Diversity (H_Y):** The haplotype diversity of Y-chromosome haplogroups within each cemetery. In patrilocal systems, related males (sharing a Y-haplogroup) remain in place, reducing local diversity. In matrilocal systems, males immigrate from various locations, increasing local diversity.

**Stat 2: mtDNA Diversity (H_mt):** The haplotype diversity of mtDNA haplogroups. The opposite pattern is expected: patrilocality involves female immigration, increasing local mtDNA diversity, while matrilocality involves females staying put, reducing it.

**Stat 3: Inter-site Differentiation (Fst_Y):** The genetic differentiation between cemeteries based on Y-chromosome haplogroups, measured using Wright's F-statistic (F_ST = (H_T - H_S) / H_T). Patrilocality should lead to distinct paternal lineages becoming dominant in different locations, increasing differentiation between sites.

**Stat 4: Inter-site Differentiation (Fst_mt):** The genetic differentiation between cemeteries based on mtDNA. Matrilocality is expected to increase differentiation in the maternal line between sites.

**Stat 5: Kin-Dyad Counts:** The absolute counts of specific first and second-degree relative pairs found within each simulated cemetery (e.g., number of father-son pairs, mother-daughter pairs, brother pairs). Patrilocal systems are expected to generate a higher frequency of co-buried patrilineal kin (father-son, brothers), while matrilocal systems should produce more co-buried matrilineal kin (mother-daughter, sisters).

## Probabilistic Assessment: Quantifying the Evidence for Inheritance Patterns

### The Need for a Formal Framework
A simple visual comparison between the summary statistics from the model and the real data is insufficient due to the model's stochastic nature. A formal statistical framework is required to quantify which of the competing scenarios is most probable given the observed archaeological evidence.

### Approximate Bayesian Computation (ABC)
Approximate Bayesian Computation (ABC) is a class of statistical methods ideally suited for this task. It allows for parameter estimation and model selection for complex, stochastic simulations where the mathematical likelihood function is intractable. The ABC process proceeds as follows:

1. **Prior Distribution:** A prior probability is assigned to each of the three scenarios (e.g., a uniform prior where each scenario has a 1/3 probability).

2. **Simulation:** The ABM is run many thousands of times. In each run, a scenario is chosen according to the prior probabilities, and the model is executed. The resulting summary statistics vector is stored.

3. **Comparison:** The Euclidean distance is calculated between the summary statistics vector from the real data and the vector from each simulation run.

4. **Rejection and Posterior Approximation:** Only the simulation runs whose summary statistics are "close" to the observed statistics (i.e., the distance is below a small tolerance, ε) are accepted. The posterior probability of each scenario is then approximated by its frequency among the accepted runs. For instance, if 70% of the accepted runs were generated under the "Mixed System" scenario, this scenario is assigned a posterior probability of 0.7.