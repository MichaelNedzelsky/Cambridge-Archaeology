# CLAUDE.md - AI Assistant Instructions

## Project Overview

This is the **Cambridge Archaeology Inheritance Pattern Analysis** project - an agent-based model using Approximate Bayesian Computation (ABC) to test 5 inheritance systems against Roman-era aDNA data from Cambridgeshire sites.

## Essential Reading

**BEFORE starting any work, READ THESE FILES:**

1. **`README.md`** - Complete project documentation with:
   - Model architecture and implementation details
   - ABC methodology explanation with concrete examples
   - Full results for all 9 archaeological sites
   - Academic references and methodological justification

2. **`description.md` (LOCAL ONLY)** - Project background containing:
   - Original problem statement and team information
   - Theoretical foundations for inheritance systems
   - Mathematical formulas for haplotype diversity
   - Archaeological context and data sources
   - **Key directive: "Ignore migration" (line 44)**

## Current Implementation Status

### ✅ **COMPLETED COMPONENTS**

1. **Data Preprocessing** (`data_preprocessing.py`)
   - Site-specific data cleaning and standardization
   - Kinship relationship processing
   - Statistical summary generation

2. **Agent-Based Simulation** (`agent_simulation.py`)
   - Individual-based population modeling over 4 generations
   - 5 inheritance systems: strongly/weakly patrilineal/matrilineal + balanced
   - **NO MIGRATION** - closed population assumption
   - Realistic burial and aDNA success rates

3. **Statistical Framework** (`inheritance_statistics.py`)
   - Nei's haplotype diversity calculation: `H = (n/(n-1)) * (1 - Σp²)`
   - Kinship pattern analysis (father-son, mother-daughter ratios)
   - Sex bias metrics and population structure analysis

4. **ABC Model Selection** (`hypothesis_testing.py`)
   - 500 simulations per site (100 per inheritance system)
   - Weighted Euclidean distance calculation
   - 5% acceptance rate for posterior probability calculation
   - Bayes factors for model comparison

5. **Complete Analysis Pipeline** (`run_full_analysis.py`)
   - Automated analysis of all 9 sites
   - Visualization generation
   - Comprehensive reporting

### 📊 **KEY RESULTS ACHIEVED**

**Analyzed 9 Roman-era Cambridgeshire sites:**
- **3 sites**: Strongly matrilineal (Arbury, Fenstanton-Cambridge Road, Fenstanton-Dairy Crest)
- **3 sites**: Strongly patrilineal (Duxford, Knobbs 2, Northwest Cambridge)
- **1 site**: Weakly patrilineal (Knobbs 3)
- **1 site**: Weakly matrilineal (Vicar's Farm)
- **1 site**: Balanced (Knobbs 1)

**Evidence quality:** 4 sites with weak evidence, 5 inconclusive (no strong evidence due to small sample sizes)

## Key Technical Details

### **ABC Weighted Distance Calculation**
```python
weights = {
    'y_diversity': 2.0,           # High - primary discriminator
    'mt_diversity': 2.0,          # High - complementary genetic signal
    'prop_father_son': 1.5,       # Medium - direct kinship evidence
    'prop_mother_daughter': 1.5,  # Medium - direct kinship evidence
    'sex_ratio': 1.0,             # Standard - supporting demographic info
    'prop_y_matches': 1.0,        # Standard - quality control
    'prop_mt_matches': 1.0,       # Standard - quality control
    'prop_inheritors': 0.5,       # Low - model artifact
}
```

### **Inheritance System Probabilities**
- **Strongly patrilineal**: 90% male, 10% female inheritance
- **Weakly patrilineal**: 70% male, 30% female inheritance
- **Balanced**: 50% male, 50% female inheritance
- **Weakly matrilineal**: 30% male, 70% female inheritance
- **Strongly matrilineal**: 10% male, 90% female inheritance

### **Files NOT in Repository (Local Only)**
These files exist locally but are ignored by git:
- `weight_explanation.py` - Weight methodology demonstration
- `description.md` - Project notes and background
- `inheritance_differences.py` - Model differences demonstration
- `analysis_summary.py` - Implementation summary
- `*.pkl` - Binary simulation results (can be regenerated)

## Common Tasks & Commands

### **Run Complete Analysis**
```bash
python run_full_analysis.py
```
Generates: CSV results, PNG visualizations, text reports, .pkl simulation data

### **Test Individual Components**
```bash
python data_preprocessing.py        # Test data loading
python inheritance_statistics.py    # Test statistical analysis
python agent_simulation.py         # Test single simulation
python hypothesis_testing.py       # Test ABC framework
```

### **Regenerate Documentation**
If you modify core functionality, update:
- `README.md` - Main documentation
- Code docstrings for key functions
- This `CLAUDE.md` file if architecture changes

## Important Notes

### **Methodological Foundations**
- **ABC rejection sampling** with strong academic backing (20+ years of literature)
- **Closed population model** - no external migration
- **Weighted statistics** prioritize genetic diversity (weight 2.0) over demographics (weight 1.0)
- **5% acceptance rate** ensures rigorous model selection

### **Limitations Acknowledged**
- Small sample sizes limit statistical power
- Temporal resolution cannot distinguish changes over time
- Burial bias affects which individuals appear in data
- Closed population assumption may not reflect reality

### **Repository Philosophy**
- **Core implementation only** in public repository
- **Analysis/demo scripts** kept local for development
- **Generated results** not version controlled (can be recreated)
- **Academic references** included for methodological justification

## Data Sources

- **Primary**: Scheib et al. (2024) - "Low Genetic Impact of the Roman Occupation of Britain in Rural Communities"
- **Kinship data**: Cambridge aDNA summary with identified relationships
- **9 sites**: Arbury, Duxford, Fenstanton (2 locations), Knobbs (3 locations), Northwest Cambridge, Vicar's Farm

## Quick Status Check

The project is **COMPLETE and FUNCTIONAL**:
- ✅ All 9 sites analyzed with rigorous ABC methodology
- ✅ Comprehensive documentation with academic references
- ✅ Clean repository focused on core implementation
- ✅ Reproducible results with statistical backing
- ✅ Publication-ready methodology and findings

**For any new analysis, start by reading README.md and description.md to understand the full context!**