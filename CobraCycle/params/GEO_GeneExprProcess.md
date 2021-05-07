### Expression data processing protocol

**Seth Rhoades**

**Overview** : This markdown highlights procedures of GEO data querying and preprocessing, for the purpose of setting lower and upper bounds on reactions in metabolic flux analysis. Any expression dataset which is fed to MetabolicEP should include a pointer to this file in the experimental conditions dictionary/json/etc, for future reference on the procedures taken.

#### 1) Read experiment files:

A GEO dataSet (GSExxx) is queried and downloaded, if the series_matrix.txt and annot.gz files exit for the expression values and platform probe IDs, respectively. These query functions heavily rely on formatting within the url of these studies and platforms (dependent on regex from requests.get()). Download links on these pages are executed to pull the expression and platform files to disk. Datasets are required to contain a valid organism ID, and at least 4 samples in the study. Under the hypothesis that expression more closely follows a power distribution then a bounded/exponential, a normalization check is applied to the expression values to ensure that the data is *not* normally distributed (with a skew cutoff,default of 0.25).

#### 2) Data processing and normalization

Expression data is processed and normalized through a sequential series of steps. Missingness thresholds are first applied, to ensure the data is sufficiently dense. A default setting is a probe must have a value in at least 75% of samples, and that a sample must have at least 75% of all possible probes detected. Next, missing data is imputed, with the median value of a probe serving as the default setting. Given the uncertainty of how any given experiment was processed, log transformation is applied, only if skew is maintained above the cutoff applied under step 1) after the log-transform, otherwise the data will be too "normal". Outliers are then removed using PCA, where any observations outside the Hotellings on the first 2 PCs are discarded. Samples are considered valuable: thus, the default setting is to only remove observations outside 3 SDs (99.7 percentile). Probes are then removed which exhibit a high coefficient of variation. A rather high value (200) is chosen as a default, given some studies may have more than 2-3 experimental groups, thus one would expect inflated CVs with real biological significance. The penultimate step, before scaling on the metabolic model bounds, is scaling the expression to preserve both the asymmetric distribution of the expression values, and the dispersion of a given probe's expression across samples (i.e. differential expression). The balance of these two factors necessitated a new scaling invention, where median-based scaling is performed (by default) across samples; however to maintain the probe-to-probe dispersion, the median-scaling is weighted by a coefficient (empirically chosen at 4). Higher coefficients correspond to higher emphasis on differential expression. This approach is similar to a MAD scaling. Finally, the expression values are placed on the scale of the metabolic model flux bounds (typically [0, 1000]). In order to provide a "complete" gene expression dataset to the metabolic model, a Gamma distribution is fit to the expression values, to provide a means to impute missing data during the gene --> reaction mapping steps before modeling metabolic flux.

### 2a) Updated processing

Design of experiments' optimization revealed some changes to the settings in 2) which improved classification performance on EP results. The coefficient of variation is instead set to 300, from 200 originally. The coefficient for scaling is set to 2, from 4 originally, on axis = 0. The sampling of expression from the Gamma distribution is no longer used to generate missing expression values. Skew is kept at 0.25, and aggregation of genes-reactions is kept at maximum expression. 

#### 3) Read metabolic model

The genome-scale metabolic model is read, based on the organism and model reference dictionary (which contains pointers to the files which correspond to the metabolic model of interest). Models are read through either cobra, or ssbio functions.

#### 4) Map probes to genes, and genes to reactions

As multiple probes may map to the same gene, an aggregation approach is taken. The default setting is to choose the maximum expression value amongst the probes which map to one gene. The gene ID, of which many types may exist on the platform file, is empirically chosen for each organism of interest to maximize the number of matches to the metabolic model. Defaults for each organism are specified in the model reference dictionary. After aggreagting to genes, then the expression value for the "entire gene" is matched to the reaction(s) in which that gene participates. If genes on the metabolic model are missing from the expression dataset, then the default behavior is to generate synthetic expression values based on the Gamma distribution fit in step 3). One strong rationale here is that reactions should not be simply turned off, nor set to a high maximum bound, simply because it isnt in the expression dataset (one would not be sure if it was either too lowly expressed to be detected, which would then be captured by the probability in the Gamma distribution anyways, or is simply not part of the expression platform, which would be unlikely if its a gene that would be expected to be normally highly expressed). 

#### 5) Constrain metabolic model, sample by sample

After gene expression processing and imputation, such that all reactions on the metabolic model can be constrained, then lower and upper bounds are set. Model bounds are first set to a default range of [-1000, 1000] for consistency across models, then constrained through choosing the maximum (default) gene expression value amongst genes which map to a given reaction (a reaction may have multiple genes which participate). As there is no consensus about how to predict a reaction's activity, if multiple genes act on it, this step is largely invented, without much scientific backing. Constraints are uniquely set for each sample on the GEO study. Finally, the lower and upper bounds for these reactions, and their reaction IDs, can be passed to metabolic flux modeling.

### 5a) Updated constraints

Optimization procedures revealed benefits from constraining exchange reactions, which generally go unconstrained from purely gene expression. AddConstraints is a new parameter to bind these exchanges, specified by idType == [ exchanges ], and default bounds at [-750, 750]. Other settings appeared to make little difference, such as the scaling applied to overlapping constraints, and how multiple constraints would be aggregated. 

### 5b) Updated constraints

Optimization procedures revealed benefits from constraining reactions where a kcat value could be matched from published values in E coli. Most of these new parameters appeared to make little difference. However, addKCatConstrain is a new option, which if set to True, will add these constraints. Relatively few reactions in the metabolic models have data on kcat, so the overall effects on the flux solutions are likely small. 

#### 6) Experimental condition extraction

For each sample, along with the flux bounds, experimental conditions are passed and stored with the result of metabolic flux modeling. These conditions include the sample ID, the root GSE study, organism, and others.

