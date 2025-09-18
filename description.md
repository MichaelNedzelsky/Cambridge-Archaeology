## Problem 2: Cambridge Archaeology
---
### Useful links

[Main room Zoom link](https://bath-ac-uk.zoom.us/j/96958639619?pwd=8wUZuYMb3iW5dhuH3LFhhAXNWCtDTA.1)

[Google drive](https://drive.google.com/drive/folders/1OHg9slBOjsW_j5EGhfyZ2WEXbS5w8Qyo)



---
### Team
- Bryony Moody, University of Sheffield, bryony.moody@sheffield.ac.uk
- Jess Enright, University of Glasgow, jessica.enright@glasgow.ac.uk
- James Andrews, University of Birmingham, j.w.andrews@bham.ac.uk
- Leyla Ranjbari, University of Birmingham, l.ranjbari@bham.ac.uk
- Matteo Sommacal
- Javier Rivas, University of Bath, j.rivas@bath.ac.uk
- Michael Green
- Nancy Nichols, University of Reading
- Faidra Katsi, University of Nottingham, faidra.katsi1@nottingham.ac.uk
- Charles Cayzer, Historic England (Analytics team), charles.cayzer@historicengland.org.uk
---

### Availability
- Jess: available for all day on the 17th, only before 10 and after 15:00 on the 18th (Boo mandatory REF meeting! :cry:), afternoon on the 19th
- James: available except 15:00-16:00 on 17th, all of 18th.
- Leyla: not available on 18 Sep.
- Faidra: not available morning on the 18th 
---

### Introduction & Links
This document serves as a workspace for the VSG on Mathematics in Archaeology and the Study of Museum Collections. Some reference information is listed beloow:

The VSG website is at https://imibath.ac.uk/2025/07/30/mathematics-in-archaeology/.

The main mural link is https://app.mural.co/t/kesummer254996/m/kesummer254996/1757412296651/90521b52864c5a7cbc31edb603d5d5e81f8558c0?sender=u7c2caed459fcc19a3f555822.

Resources like papers, relevant tutorials, and small datasets can be found at https://drive.google.com/drive/folders/1OHg9slBOjsW_j5EGhfyZ2WEXbS5w8Qyo.

A link to large datasets will be added at a later date. Please check data sizes before trying to download an entire folder.

#### Notes from the presentation
- Do we have a model of movement of the population around the country at that time, will migration need to be incorportated into the model to account for people moving away meaning that data is not missing because due to migration they would have never been there in the first place. <span style="color:blue">Ignore migration.</span>
- Do we have any overall information of geographic distribution and frequency of haplotypes? <span style="color:blue">All over Europe not local at all.</span>
- Do we have a relative chronology based on relatedness and the degree of relatedness that might give us more information about the model? - <span style="color:blue">Potentially could be interesting as it would show gaps but many gaps will be becuase of people marrying off.</span>
- Do we need to take into account a degree of certainty for the kinship or can we assume it to be definitive? - <span style="color:blue">Take them as 100% true. Ignore anything after the dash in the haplogroups</span>
- Burial vs inhumation: what is the difference? Does it affect anything we need to model? <span style="color:blue">- Burial is any way of disposing humans, inhumation is burial on.</span> 
- Does geographical proximity to known Roman settlements mean that inheritance practices might be more likely to be influenced? <span style="color:blue">In Sardina that wasn't the case so something to take into account for this point. - In really rural areas such as NW, everything is close to a roman settlement, all within 20 miles. Arbury might be a slightly different case due to a higher status of the occupants</span>
- What is the youngest age we might expect someone to inherit? <span style="color:blue">15-16 onwards adult vs non adult in the data.</span>
- Does age/gender impact the how deteriorated the DNA is? <span style="color:blue">We don't know but assume no? Enviromental factors are more important.</span>
- What does the colour in the tables represent? <span style="color:blue">Nothing for us to worry about.</span>
- Is it at all meaningful to try and model how people are related to each other giving the missing information?
- Do we want to incorporate information about how close the burials were to each other to help model likelihood of being related? <span style="color:blue">- The issue is the time passed may mean that relatives might not be buried closer to each other.</span>
- Assume population max estimates are uniform distributed, population max means at any given one time.
- Suspect no influence from the Romans on inheritance pre-occupation

#### What would a matrilocal society look like in the burial record and how would that differ from a patrilocal?
 ##### Strongly matrilineal
 No men related to eachother in a cemetry but lots of women related to each other
 ##### Weakly Matrilineal
 Example of parent-daughter relationships or aunt/uncle-niece, women with the same haplogroup
 
  ##### Balanced (equal men and women)
Equal number of relationships between  older relative - younger male and older relative - younger female

 #### Weakly patrilineal
  Example of parent-son relationships or aunt/uncle-nephew
 
 #### Strongly patrilineal
No women related to eachother in a cemetry but lots of men related to each other

#### Definition of variables
##### Knowns
- For some of the burials we know kinship (not random)
- Max population at any given time, we are assuming this is uniform (random variable)
- Age of the samples (roughly)
- How many individuals have been sampled for aDNA
- Exclusion of relations based on haplogroups (e.g. Knobbs farm non of the women belong to the sample haplogroup)
- Cemetries typically only span for 4 generations for an upper threshhold

##### Unknowns
- How many children (if any) a given person had, for which we have DNA
- How many individuals haven't been buried
- At what ratio does women being buried with their parents indicate a matrilocal society
- How likely are non-inheriting children to move away
- How strong is the association between moving in with a woman as opposed to a woman inheriting, focus on locality first
- If they move away following marriage, how far? Would they not ever be buried back with their families. Either in a central burial. First degree marriages are common  
- Expected number of children, also if they're moving away what happens when the other partner also is not set to inherit? 

##### Model
- Null model -> some rate of people remaining differentiated by sex 
- Alt model -> some other rate of people staying 
- Key question: Can the uncertainty be explicitly defined or is it truly just unknown (for any given variable)

Monte Carlo hypothesis testing? Is the null completely partilocal i.e. no women inherit (patrilineal)

Two models to consider? Assuming aDNA is complete first. Then may move on to assuming data is missing at probability $\alpha$.

Is patrilocal equivalent to patrilineal?

#### Clustering - 19th 
Liela wants to see if she can use clustering to group into matrilocal and partilocal

#### Roman Britain Map taken from Wikipedia
![Roman.Britain.Romanisation](https://hackmd.io/_uploads/ryDfbEuilx.jpg)
[Wikipedia Source](https://en.wikipedia.org/wiki/Romano-British_culture)





#### Datasets Investigation

#### Statistical Inference




#### Scratch ideas

Null simulation model of modement in and out?  If there were a lot of coverage of sites we could build an overall network of haplotype movement and haplotype geography, but this may not be possible from the data - is it possible from any existng data?  Do we have any background information on overall geography of haplotype or frequency of haplotypes overall?


- TODO: make suggestions for null geographic haplotype modelling

## Agent based model?

### We Have

- Full data composes of sex male (M) or female (F).
- Estimated age class or estimated age.
- Uniparental assignment (mtDNA haplogroup or missing; Y haplogroup or missing)
- Pairwise kinship calls for some pairs (i.e. first/second degree, with uncertainty)
- Contextual covariates? (grave type, see Supplementary Figures for this [paper](https://academic.oup.com/mbe/article/41/9/msae168/7741671))

Not all data is complete, but a small number are complete especially for Duxford.

### Possible Attempt

Under a social model $M$, i.e. matrilocal, patrilocal, bilocal, transitional. Each $M$ can be parameterised by $\Theta$ (demography, burial-bias, adoption rates, aDNA success rates). Time over which we are interested, inheritor indicator, burial choice (is this important?), genetic markers (mtDNA, Y, autosomal genotypes). Simulate a cemetry given these assumptions.

#### Burial probability

Burial and aDNA recovery are biased processes. The probability an individual $i$ who dies at time $t$ in the settlement $s$ is:
- Probability individual $i$ is buried in the sampled site

$$p_i^{\text{Burial}}=f\left(\text{site specific baseline}, \text{age}, \text{status}, \text{Inheritor}\right)$$

- Probability of a successful assignable aDNA given individual $i$

$$p(\text{aDNA observed}|i)=p_s^{\text{aDNA}}q(\text{preservation covariates)}$$

where $q$ is the preservation-related covariates for individual $i$.

#### Uniparental haplogroup counts

#### Kinship counts

#### Summary statistics (X/mt/Y diversity and IBD [Identical By Descent])

#### Composite likelihood and posterior

#### Approximate Bayesian Computation - Sequential Monte Carlo

#### Priors and identifiability

## Kinship

### Duxford site

![image](https://hackmd.io/_uploads/B1M5S8tjlg.png)
Kinship for a family of bodies found at Duxford

![image](https://hackmd.io/_uploads/SJmDYPtilx.png)
Kinship for a family of bodies found at Duxford, not Unknown means the body was not identified in DNA samples

## Code

[Code](https://drive.google.com/drive/folders/1vbmPPi6zL3UJE8cGeaDvj9EkjFXEZhik?usp=drive_link)

**Patrilineal:**
![Patrilineal](https://hackmd.io/_uploads/rk1M5vKjel.png)

**Matrilineal:**
![Matrilineal](https://hackmd.io/_uploads/HyJM9wYoxe.png)

## Haplotypes diversity

Given a mulstiset of haplotypes with counts $c_1,\cdots,c_k$ such that $\sum_{i=1}^{K}c_i=n$ we can define, $p_i=c_i/n$ and $D=\sum_{i=1}^Kp_i^2$ then Nei's haplotype diversity is given by,
$$h=\frac{n}{n-1}(1-D)$$

## Pure uninterrupted patrilineal descent

Calculating the expected haplotype diversity in a cemetery from pure uninterrupted patrilineal descent containing only the founder, their spouse, their direct descendants, and the descendants’ spouses. This will be performed for the mitochondrial DNA and the Y-chromosome DNA. Note it is expected that mutations will be insignificant on this timescale. Furthermore it is assumed that all spouses are sufficiently distantly related that their haplotype is a novel introduction to the lineage (extreme assumption and probably not true). What is the expected $h_{mt}$ and $h_Y$? 

Starting with $h_Y$ only inheritors carry the Y-chromosome so the population is only the males found in the cemetery. Since the founder had the Y-chromosome haplotype and passed it down to the inheritors, without extramatrital parings the only haplotype will be the founder's Y-chromosome. Hence $p_{Y,\text{Founder}}=1$ and $D_Y=1$ so, $h_Y=0$. I.e. there is no diversity expected in the Y haplotypes. 

For the mitochondrial DNA, it is expected that with the exception of the founder's mitochondrial DNA and the final generation's female spouse's DNA the maternal mitochondrial DNA will appear twice in the cemetery, their own and their son's. As such, there will be $g-1$ (where $g$ is the number of generations) haplotypes with count 2 and 2 haplotypes with count 1 (the male founder and the last female spouse). 

$$D_{mt}=(g-1)\left(\frac{2}{2g}\right)^2+2\left(\frac{1}{2g}\right)^2=\left[1-\frac{1}{2g}\right]\left(\frac{1}{g}\right)$$

since $2g$ represents the number of haplotypes in the cemetery. Hence the haplotype diversity for the mitochondrial DNA is given by,

$$h_{mt}=\frac{n}{n-1}\left(1-\left[1-\frac{1}{2g}\right]\left(\frac{1}{g}\right)\right)=\frac{n}{n-1}\left(1-\frac{1}{g}+\frac{1}{g^2}\right)=\frac{2(g(g-1)+1}{g(2g-1)}.$$


Patrilineal descent can be interrupted in numerous ways so this in not expected in real cemeteries.

## Pure uninterrupted matrilineal descent

Calculating the expected haplotype diversity in a cemetery from pure uninterrupted matrilineal descent containing only the founder, their spouse, their direct descendants, and the descendants’ spouses. This will be performed for the mitochondrial DNA and the Y-chromosome DNA. Note it is expected that mutations will be insignificant on this timescale. Furthermore it is assumed that all spouses are sufficiently distantly related that their haplotype is a novel introduction to the lineage (extreme assumption and probably not true). What is the expected $h_{mt}$ and $h_Y$?

Starting with $h_Y$ only the spouses of the owners will be present. Since the inheritors spouses $h_Y$ are all novel Y-haplotypes so the count for each haplotype is 1 and the size of the population is $g$ (only males).
$D_Y=g\frac{1}{g^2}=\frac{1}{g}$ and $h_Y=1$, since $n=g$.

For the mitochondrial DNA each $g$ males bring a unique mitochondrial DNA once, while each $g$ feamles contribute an identical by descent mitochondrial DNA. So, $n=2g$, $D_{mt}=\left(\frac{g}{2g}\right)^2+g\left(\frac{1}{2g}\right)^2=\frac{2(g-1)}{2g-1}$ and,

$$h_{mt}=\frac{2g}{2g-1}\left(1-\frac{2(g-1)}{2g-1}\right)=\frac{2g}{(2g-1)^2}$$

Matrilineal descent can be interrupted in numerous ways so this in not expected in real cemeteries.

## Theoretical results

![Theoretical_Linear_Mitochondrial_DNA](https://hackmd.io/_uploads/B1Glbctigl.png)



# References

[Supporting information: Continental influx and pervasive
matrilocality in Iron Age Britain](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08409-6/MediaObjects/41586_2024_8409_MOESM1_ESM.pdf)

Sources
ADNA data: 	Christiana L Scheib, Ruoyun Hui, Alice K Rose, Eugenia D’Atanasio, Sarah A Inskip, Jenna Dittmar, Craig Cessford, Samuel J Griffith, Anu Solnik, Rob Wiseman, Benjamin Neil, Trish Biers, Sarah-Jane Harknett, Stefania Sasso, Simone A Biagini, Göran Runfeldt, Corinne Duhig, Christopher Evans, Mait Metspalu, Martin J Millett, Tamsin C O’Connell, John E Robb, Toomas Kivisild, Low Genetic Impact of the Roman Occupation of Britain in Rural Communities, Molecular Biology and Evolution, Volume 41, Issue 9, September 2024, msae168, https://doi.org/10.1093/molbev/msae168 

Includes extensive Supplementary data, including all aDNA results. 

Arbury:	Fell, C. I. (1956). Roman Burials found at Arbury Road, Cambridge, 1952. Proceedings of the Cambridge Antiquarian Society 49. Vol 49, pp. 13-24. https://doi.org/10.5284/1072909

Duxford: 	Lyons  A. Life and Afterlife at Duxford, Cambridgeshire: archaeology and history in a chalkland community. 2011. Report No 141. https://eaareports.org.uk/publication/report141/ 

North West Cambridge. Unpublished. Grey literature: Cessford   C, Evans   E. North West Cambridge Archaeology. University of Cambridge 2012-2013 Excavations. 2014. Report No. 3: Parts 2: pages 347f. https://doi.org/10.17863/CAM.101137. 

Fenstanton. Unpublished. 

Knobb’s Farm: Wiseman   R, Neil   B, Mazzilli   F. Extreme justice: decapitations and prone burials in three late Roman cemeteries at Knobb's Farm, Cambridgeshire. Britannia. 2021:52:1–55. DOI: https://doi.org/10.1017/S0068113X21000064.

Vicar’s Farm: Evans   C, Lucas   G. Hinterlands & inlands: the archaeology of west Cambridge and Roman Cambridge revisited. Cambridge: McDonald Institute for Archaeological Research; 2020.

Thompson, E.A., 2000. Statistical inference from genetic data on pedigrees. IMS.