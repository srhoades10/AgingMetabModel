import gzip, os, requests, urllib, re, warnings, cobra, libsbml, json, csv, random
import pandas as pd
from scipy import stats
import numpy as np
from ssbio.io import load_json as load_gem_json
from itertools import chain
import subprocess

def experimentFilesReader(experimentName, geoDir = 'Cobracycle/GEOData', 
    skewLimit = 0.25, expandPlatformID = False, sep = '///', 
    checkImportNormalization = True, **params):
    """ Query a study from GEO, read in the series_matrix.txt data and the annot.gz
        annotation table for the transcript IDs as a data frame. 
    
        Args: 
            experimentName: Str - GSExxx experiment ID from GEO.
            params: Dict: Parameters from .yaml
            geoDir: Str - Directory to store annot and series_matrix files.
            skewLimit: Float - Minimum skew value of the dataset to accept for
                further processing. Used as a filter against too "normal" of data.
            expandPlatformID: Boolean - Expand the platformID column with 
                expandPlatformIDs(). Currently False, as its better to use it
                later in the process.
            sep: Str - Separator for multi-gene entries on the str.split()
            checkImportNormalization: Boolean - Perform a check on whether the
                expression data meets a skewness threshold
                
        Returns:
            data: DataFrame - GSE expression data table
            platform: DataFrame - Platform annotations associated with the GSE
            organism: Str - Organism name from GEO
    """
    experimentFile = '{0}_series_matrix.txt.gz'.format(experimentName)

    annotationFile = downloadExpFiles(experimentName, geoDir = geoDir)
    platform = annotationReader(annotationFile, geoDir = geoDir)

    counter = 0
    with gzip.open('{0}/{1}'.format(geoDir, experimentFile), 'rt') as f:
        header = f.readlines()
    for row in header:
        counter += 1
        if '!series_matrix_table_begin' in row:
            finalcount = counter
        if '!Sample_organism' in row:
            findOrg = re.findall('\"(.*)\"', row)
            orgSplit = re.split('\"\\t\"', findOrg[0])
            organism = list(set(orgSplit))[0]
       
    data = pd.read_csv('{0}/{1}'.format(geoDir, experimentFile), 
        compression = 'gzip', sep = '\t', skiprows = finalcount, 
        low_memory = False)

    if len(data) < 2:
        raise Exception('Invalid data matrix')
    
    if len(data.columns) < 5:
        raise Exception('Less than 4 samples present in this study, not suitable '
             'for future analysis')
    
    if 'organism' not in locals():
        raise ImportError('Organism ID for this GEO study not found')
    
    if checkImportNormalization == True:
        checkNormalization(data, skewLimit = skewLimit)
    
    data = data.drop(len(data) - 1)

    return data, platform, organism


def downloadExpFiles(experimentName, geoDir = 'CobraCycle/GEOData'):
    """ Download the series matrix file from GEO by the accession number. Then 
        also download the platform annotation file for transcript IDs. Note these
        urls are hard-coded based on GEO queries (hopefully these won't change!).
        
        Args: 
            experimentName: Str - GSExxx experiment ID from GEO. Inherited from
                experimentFilesReader()
            experimentFile: Str - GSExxx experiment file from GEO. Inherited from
                experimentFilesReader()
            geoDir: Str - Directory to store annot and series_matrix files.

        Returns:
            annotationFile: Str - Platform file associated with the GSE experiment.
            The series_matrix.txt.gz for the GSE experiment is written to file
    """
    url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={0}'.format(experimentName)
    urlget = requests.get(url)

    if 'Series Matrix File(s)' in urlget.text:
        fileName = '{0}_series_matrix.txt.gz'.format(experimentName)
        fullFile = '{0}/{1}'.format(geoDir, fileName)
        if fileName not in os.listdir(geoDir):
            link = re.findall('href=\"(.*)\/\".*Series Matrix File', urlget.text)[0]
            try:
                urllib.request.urlretrieve(link + '/' + fileName, fullFile)
            except urllib.error.URLError:
                raise Exception('Unable to retrieve file, it may not be public')

        if 'Platforms' in urlget.text:
            annotationFile = downloadAnnotation(urlget.text, geoDir = geoDir)
            if annotationFile not in os.listdir(geoDir):
                raise Exception('Annotation file for this experiment found but '
                    'download unsuccessful')
            return annotationFile

        else:
            os.remove(fullFile)
            raise Exception('No platform annotation reference file could be found')
    else:
        raise Exception('No .txt matrix file present for this experiment') 


def downloadAnnotation(experimentURLText, geoDir = 'Cobracycle/GEOData'):
    """ Parse the url requests get() from a given GSE experiment and download the
        annotation file. Inherits from downloadExpFiles. Note these string 
        searches are hard-coded based on GEO queries (hopefully these won't 
        change!).
    
        Args:
            experimentURLText: String - results of the requests.get() for a GSE.
                experiment. Inherits from downloadExpFiles()
        Returns:
            fileName: String - Filename of annotation file for the GSE experiment.
                Note that the file itself is downloaded to, but the file name is
                used for an os.listdir() check later
    """
    cleanText = re.sub('<.*?>|\\n', ' ', experimentURLText)
    platformID = re.findall('Platforms \(\d+\)       (\w+) ', cleanText)[0]
    fileName = '{0}.annot.gz'.format(platformID)
    if fileName in os.listdir(geoDir):
        return fileName

    else:
        url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={0}'.format(platformID)
        urlget = requests.get(url)

        if 'Annotation SOFT table..' in urlget.text:
            fullFile = '{0}/{1}'.format(geoDir, fileName)
            link = re.findall("OpenFTP\(\'(.*{0})".format(fileName), urlget.text)[0]
            urllib.request.urlretrieve(link, fullFile)
            return fileName
        elif 'HiSeq' in urlget.text:
            raise Exception('Possible RNASeq experiment detected')
        else:
            raise Exception('No platform annot.gz file present for this experiment') 


def annotationReader(annotationFile, geoDir = 'CobraCycle/GEOData'):
    """ Given a platform annotation file, convert the .gz to a DataFrame """        
    
    fullFile = '{0}/{1}'.format(geoDir, annotationFile)
    counter = 0
    with gzip.open(fullFile, 'rt') as f:
        header = f.readlines()
    for row in header:
        counter += 1
        if '!platform_table_begin' in row:
            finalcount = counter
    
    platformDF = pd.read_csv(fullFile, compression = 'gzip', sep = '\t', 
        skiprows = finalcount, low_memory = False)
    if len(platformDF) < 2:
        raise Exception('Invalid platform matrix')

    platformDF = platformDF.drop(len(platformDF) - 1)
    return platformDF


def checkNormalization(exprData, skewLimit):
    """ After reading in the expression data table, check for data skewness.
        We actually want skewed data which isn't centered and scaled (although
        log2 transformation is ok). skewLimit = 0.25 is empirically chosen. """

    dataOnly = exprData[exprData.columns[1:]]
    dataValues = np.ravel(dataOnly)
    dataValues = dataValues[~np.isnan(dataValues)]
    skew = stats.skew(dataValues)

    if skew < skewLimit:
        raise Exception('Data appears roughly normally distributed and was '
            'likely centered')
    else:
        return


def fitGamma(exprData, dropCol = 'ID_REF', minLoc = 0., **params):
    """ Fit a gamma distribution to the expression data, and return the shape and
        location parameters. To be used later in the pipeline for sampling missing
        expression data 
        
        Args:
            exprData: DataFrame - Expression data table
            params: Dict - Parameters from .yaml
            dropCol: Str - Column to exclude during Gamma fit
            minLoc: Float - Min location of Gamma

        Return:
            shape, loc: Float - Gamma distribution moments
    """
    if dropCol not in exprData.columns:
        raise Exception('Probe/gene ID column not found in expression data')

    dataOnly = exprData.drop(dropCol, axis = 1)
    shapeVal, locVal, _ = stats.gamma.fit(np.ravel(dataOnly))
    if locVal < minLoc:
        locVal = minLoc
    return shapeVal, locVal


def readMetabolicModel(organism, modelDict, modelPath = 'ref', 
    preferredWorm = 'WormFlux', **params):
    """ Read in metabolic model using cobrapy or ssbio functions. Extract the set
        of gene IDs from the model, which may need to be modified in order to 
        match to gene IDs found in GEO platform gene IDs. Thus a gene set, and a
        dictionary of the original metabolic model gene IDs to the modified IDs
        are created, which is necessary later in the process in order to constrain
        the model. The gene ID which provides the best match between a platform
        file and a given metabolic model was experimentally-derived, and embedded
        in the modelDict. The original flux bounds for every reaction is also
        stored as a dictionary, in order to reset the model bounds later in the
        process without having to re-read the model object.
    
        Args:
            organism: String - Organism name
            modelDict: Dict - Dictionary of human-readable models to file names.
                Created manually outside of this function.
            params: Dict - Parameters from .yaml
            modelPath: String - Folder containing metabolic models (assume .XMLs)
            preferredWorm: String - Pick between WormFlux and C elegans models

        Returns:
            modelObject: Object - CobraPy or ssbio model
            modelGeneSet: Set - Genes from the model object, from 
                extractModelGeneList()
            geneIDMatch: String - Column ID from the expression data used for 
                joining and constraining. Derived empirically (see 10/10/2018 
                    lab notebook)
            geneDict: Dict - Dictionary of gene IDs from the metabolic model, 
                from extractModelGeneList().
            boundDict: Dict - Dictionary of original flux bounds, used for
                resetting boudns between samples.
    """
    modelDictOrgs = [modelDict[x]['Organism'] for x in modelDict]
    if organism not in modelDictOrgs:
        raise ValueError('Metabolic model for {0} not available, try one of '
            '{1}'.format(organism, modelDictOrgs))

    if organism == 'Caenorhabditis elegans':
        model = preferredWorm
    else:
        model = [x for x in modelDict if modelDict[x]['Organism'] == organism][0]

    modelFile = modelDict[model]['File']
    geneIDMatch = modelDict[model]['IDMatch']
    cwd = os.getcwd()

    filePath = '{0}/{1}'.format(cwd, modelPath)
    fullPath = '{0}/{1}/{2}'.format(cwd, modelPath, modelFile)
    
    if modelFile not in os.listdir(filePath):
        raise ImportError('Model file not found. Are you specifying the right path?')

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            print('Cobra warnings likely occured due to malformed gene reaction '
                'rules or metabolites appearing as reactants and products, however '
                'these will be suppressed here to avoid excessive printouts')
            if 'mat' in str(modelFile):
                modelObject = cobra.io.load_matlab_model(fullPath)
            else:
                modelObject = cobra.io.read_sbml_model(fullPath)
            modelGeneSet, geneDict = extractModelGeneList(modelObject)
            boundDict = extractModelBounds(modelObject)

        if len(modelGeneSet) == 0:
            raise ImportError('Cobra model has no genes!')
        else:
            print('{0} read-in successful'.format(model))
            return modelObject, modelGeneSet, geneIDMatch, geneDict, boundDict

    except cobra.io.sbml3.CobraSBMLError: #try ssbio read from compressed json
        print('Metabolic model not compatible with CobraPy, will try to load '
            'with ssbio')
        try:
            modelObject = load_gem_json(fullPath, decompression = True)
            modelGeneSet, geneDict = extractModelGeneList(modelObject = modelObject)
            boundDict = extractModelBounds(modelObject)
            
            if len(modelGeneSet) == 0:
                raise ImportError('Cobra model has no genes!')
            else:
                return modelObject, modelGeneSet, geneIDMatch, geneDict, boundDict
        except:
            raise ImportError('Metabolic model file found, but failed to read.')


def extractModelBounds(modelObject):
    """ Extract the metabolic model flux bounds as a dictionary """

    boundDict = dict()
    for rxn in modelObject.reactions:    
        boundDict[rxn.id] = {'Lower bound': modelObject.reactions.get_by_id(rxn.id).lower_bound, 
            'Upper bound': modelObject.reactions.get_by_id(rxn.id).upper_bound}
    
    return boundDict


def extractModelGeneList(modelObject):
    """ Extract gene IDs from the metabolic model. Inherits from 
        readMetabolicModel(). See readMetabolicModel() for args and returns. """

    modelGeneSet = set([x.id for x in modelObject.genes])
    
    geneDict = dict()
    for gene in modelGeneSet:
        geneDict[gene] = gene

    if len(modelGeneSet) == 0:
        raise ValueError('No genes able to be extracted from the metabolic model')
    
    return modelGeneSet, geneDict


def aggregateProbesToGenes(exprData, platform, aggregateID, expandID = True, 
    sep = '///', aggregateFun = 'max', **params):
    """ Merge probe-level expression data with its platform IDs. Assume expression 
        data contains ID_REF and platform contains ID (standard in GEO). Genes
        which map to multiple probes are aggregated by taking the max (default) 
        value amongst those probes. Aggregate ID would come from findMaxIDMatch(),
        or specified in the model dictionary created outside this function, which
        resulted from experimental observation of the maximal number of ID matches
        (findMaxIDMatch() is slow, so we don't want to run it unless we have to).
        This aggregation is a last step before matching to metabolic model genes.
        
        Args:
            exprData: DataFrame - Expression data
            platform: DataFrame - Platform annotation file for expression array.
            aggregateID: String - Column ID for aggregating the probes
            params: Dict - Parameters from .yaml
            expandID: Boolean - Expand the aggregate ID column if it contains 
                multiple genes in a single cell which are separated by a string
            sep: String - Separator for string split in the expandID call.
            aggregateFun: String - Function for aggregating probes to the same
                gene id. Can take the values of max, min, mean, median 

        Returns:
            exprDataAgg: DataFrame - exprDataMerge aggregated by the specified ID.        
            sampleList: List - List of sample IDs (note its exported here, as 
                opposed to at the initial query, since samples may have been dropped
                during dataProcess()).
    """
    if 'ID' not in platform.columns or 'ID_REF' not in exprData.columns:
        raise Exception('IDs not in platform file, or ID_REF not in expression data')
    
    if aggregateID not in platform.columns:
        raise ValueError('Aggregation ID not found in list of platform IDs')
    
    if aggregateFun not in ['max', 'min', 'mean', 'median']:
        raise ValueError('Invalid aggregation function, try "max", "min", "mean", '
            'or "median"')

    sampleNames = [x for x in exprData.columns if x != 'ID_REF']
    if not all([re.search('GSM', x) for x in sampleNames]):
        raise Exception('Sample names not properly extracted from processed '
            'expression data')

    platform = platform[platform[aggregateID].notnull()] #avoid aggregating NAs
    if len(platform) == 0:
        raise Exception('All values in the platform gene ID columns were NA')

    if expandID == True:
        if any(platform[aggregateID].astype(str).str.contains(sep)):
            platform = expandPlatformIDs(platform = platform, 
                colName = aggregateID, sep = sep)        

    exprDataMerge = exprData.merge(platform, how = 'inner', left_on = 'ID_REF', 
        right_on = 'ID')

    if len(exprDataMerge.columns) == len(exprData.columns):
        raise Exception('Gene IDs not successfully attached to expression data')
    
    trimCols = [aggregateID] + sampleNames
    dataTrim = exprDataMerge[trimCols]
    
    if aggregateFun == 'max': #Wordy code but faster results
        dataGroup = dataTrim.groupby(by = aggregateID).apply(lambda x: np.max(x,
             axis = 0))
        dataGroup[aggregateID] = dataGroup.index
    if aggregateFun == 'min': 
        dataGroup = dataTrim.groupby(by = aggregateID).apply(lambda x: np.min(x,
             axis = 0))
        dataGroup[aggregateID] = dataGroup.index
    if aggregateFun == 'mean': 
        dataGroup = dataTrim.groupby(by = aggregateID).apply(lambda x: np.mean(x,
             axis = 0))
        dataGroup[aggregateID] = dataGroup.index
    if aggregateFun == 'median': 
        dataGroup = dataTrim.groupby(by = aggregateID).apply(lambda x: np.median(x,
             axis = 0))
        dataGroup = dataGroup.apply(pd.Series) #Median gives arrays back not DataFrame
        if len(dataGroup.columns) != len(sampleNames):
            if len(dataGroup.columns) == len(sampleNames) + 1:
                dataGroup = dataGroup[dataGroup.columns[1:]]
            else: 
                raise Exception('Median-aggregation failed due to groupby() mismatch '
                    'with sample names')
        dataGroup.columns = sampleNames
        dataGroup[aggregateID] = dataGroup.index

    exprDataAgg = dataGroup.merge(platform, how = 'inner', left_index = True, 
        right_on = aggregateID)

    uniquePre = len(set(exprDataMerge[aggregateID][exprDataMerge[aggregateID].notnull()]))
    uniquePost = len(set(exprDataAgg[aggregateID][exprDataAgg[aggregateID].notnull()]))
    
    if uniquePre != uniquePost:
        raise Exception('IDs not successfully re-merged to expression data after '
            'aggregation')

    return exprDataAgg, sampleNames


def expandPlatformIDs(platform, colName, sep):
    """ Platform annotations may contain multiple IDs in a single cell, commonly
        separated by ///. Expanding the annotations by splitting on this separator
        may increase the number of matches to the metabolic model IDs.

        Args:
            platform: DataFrame - Table containing platform IDs (may or may not be
                joined to the expression data itself)
            colName: String - Column to split and expand the entire dataframe 
                (may be derived from findMaxIDMatch)
            sep: String - Separator for the multi-IDs, commonly triple slash.
        
        Returns:
            expandPlatform: DataFrame - Expanded exprData via str.split() on
                column of interest
    """
    if colName not in platform.columns:
        raise ValueError('Column name not found in the annotation table')

    origColLen = len(platform.columns)
    annotationSplit = platform[colName].astype(str).str.split(sep).apply(np.array, 1)
    
    fillIndex = [] #Loop over np array is 10x faster than apply(pd.Series).stack()
    fillID = []
    for row in range(len(annotationSplit)):
        try:
            for item in annotationSplit.iloc[row]:
                fillIndex.append(row)
                fillID.append(item)
        except:
            fillIndex.append(row)
            fillID.append(np.nan)

    aggDF = pd.DataFrame()
    aggDF[colName] = fillID
    aggDF.index = fillIndex
    aggDF.name = colName
    expandPlatform = platform.drop(colName, axis = 1).join(aggDF)

    if len(set(platform['ID'])) < len(set(expandPlatform['ID'])):
        raise Exception('Probe IDs were lost during str split on gene ID column')

    if len(expandPlatform.columns) != origColLen:
        raise Exception('Number of columns after str.split expansion changed!')

    return expandPlatform


def matchCBMGenes(sampleList, modelGeneSet, exprDataWithIDs, geneID, gammaShape,
    gammaLoc, scaleVals = True, minFluxVal = 0, maxFluxVal = 1000,
    findMaxIDOnFailure = True, checkSeparation = True, sep = '///', 
    sampleMissingGenes = True, **params):
    """ Find the IDs from the platform which maximally match to the gene IDs from 
        the cobra model, then merge. This step (should be) the last step before
        constraining the metabolic model. 

        Args:
            sampleList: List - List of sample names from GEO. Derived from 
                aggregrateProbesToGenes()
            modelGeneSet: Set - Set of genes from metabolic model.
            exprDataWithIDs: DataFrame - Expression data, with probes aggregated
                to genes, and platform IDs attached (from aggregateGenesToProbes). 
            geneID: String - Column used for gene ID matching
            gammaShape: Float- Shape of gamma dist fit to the expression data 
            gammaLoc: Float - Location for gamma dist fit to expression data
            params: Dict - Parameters from .yaml
            scaleVals: Boolean - Scale synthetic expression data
            minFluxVal: - Float/Int - Minimum value in scaling
            maxFluxVal: - Float/Int - Maximum value in scaling
            findMaxIDOnFailure: Boolean - If the pre-specified platform ID fails
                to match to the metabolic model genes, run findMaxIDMatch() to
                find on the platform ID column with the maximal number of matches
                (note this is slow!)
            sampleMissingGenes: Boolean - For genes in metabolite model, but not
                in the expression data, sample values based on the power distribution
        
        Returns:
            exprDataFilter: DataFrame - Expression data matched to model genes
    """
    if len(modelGeneSet) == 0:
        raise ValueError('Invalid gene list for metabolic model for matching to '
            'expression data')
 
    origGeneLen = len(modelGeneSet)
    if (len(sampleList) == 0 or 
        len(set(exprDataWithIDs.columns).intersection(sampleList)) != len(sampleList)):
        raise Exception('Invalid sample ID list for matching to metabolic model genes')

    filterCols = [geneID] + sampleList
    exprDataFilter = exprDataWithIDs[filterCols]
    missingGenes = set([x for x in modelGeneSet if x not in list(exprDataFilter[geneID])])
    exprDataFilter = exprDataFilter[exprDataFilter[geneID].isin(modelGeneSet)]

    if findMaxIDOnFailure == True and len(exprDataFilter) == 0:
        warnings.warn('No gene IDs from the expression data match to the metabolic, '
            'model, an attempt will be made to unbiasedly scan all platform IDs '
            'for any matches to the metabolic model genes (note this may take '
            'some time)', Warning)
        geneID = findMaxIDMatch(platform = exprDataWithIDs, 
            modelGeneSet = modelGeneSet, checkSeparation = checkSeparation, 
            sep = sep)
        filterCols = [geneID] + sampleList
        exprDataFilter = exprDataWithIDs[filterCols]
        exprDataFilter = exprDataFilter[exprDataFilter[geneID].isin(modelGeneSet)]

        if len(exprDataFilter) == 0:
            raise Exception('Expression data join to metabolic model failed')

    if findMaxIDOnFailure == False and len(exprDataFilter) == 0:
        raise Exception('Expression data join to metabolic model failed')

    exprDataFilter.drop_duplicates(keep = 'first', inplace = True)
    exprDataFilter.reset_index(drop = True, inplace = True)

    if len(set(exprDataFilter[geneID])) != len(exprDataFilter):
        raise Exception('Duplicate gene IDs found in final expression data after '
            'matching with metabolic model genes, check that any given gene ID '
            'doesnt have multiple expression values within the same sample '
            'associated with it') #if so, then choose one row?
    
    print('{0} out of {1} genes in the metabolic model have matching expression '
        'data'.format(len(set(exprDataFilter[geneID])), origGeneLen))

    if sampleMissingGenes == True:
        print('... however, values for the missing genes will be sampled '
            'based on the distribution of the existing expression values')
        
        exprDataFilter = imputeExpression(exprData = exprDataFilter, 
            sampleList = sampleList, geneID = geneID, missingGenes = missingGenes, 
            gammaShape = gammaShape, gammaLoc = gammaLoc, scaleVals = scaleVals, 
            minFluxVal = minFluxVal, maxFluxVal = maxFluxVal)

        if not all(exprDataFilter.columns.isin(filterCols)):
            raise Exception('imputeExpression did not return proper IDs and sample '
                'names')
        if len(set(exprDataFilter[geneID])) != origGeneLen:
            raise Exception('Genes from the metabolic model are still missing after '
                'imputeExpression(). Check proper input parameters')

    return exprDataFilter


def findMaxIDMatch(platform, modelGeneSet, checkSeparation, sep):
    """ If we don't know which gene ID column we should match to the gene IDs in
        the metabolic model, scan over all columns and return the column ID with
        the most matches. Inherits from, and args/returns detailed in, 
        matchCBMGenes(). This function is expensive, only run if necessary
    """
    validCols = [ID for ID in platform.columns if 
        len(set(platform[ID][platform[ID].notnull()]))>0]

    if len(validCols) == 0:
        raise ValueError('All columns in the platform table appear to be NA/null')

    matchDict = dict.fromkeys(validCols)
    for col in validCols:
        if checkSeparation == True and any(platform[col].astype(str).str.contains(sep)):
            uniVals = set(chain.from_iterable([re.split(sep, 
                str(x)) for x in list(platform[col])]))
            matchDict[col] = len(modelGeneSet.intersection(uniVals))
        else:
            uniVals = set(platform[col])
            matchDict[col] = len(modelGeneSet.intersection(uniVals))
    
    maxMatch = max(matchDict.values())
    if maxMatch == 0:
        raise ValueError('No matches found from platform IDs to metabolic model '
            'genes, check that the organism from GEO matches the metabolic model')
    
    matchKeys = [key for key, value in matchDict.items() if value == maxMatch]
    if len(matchKeys) > 1:
        warnings.warn('Matching between platform IDs and metabolic model genes '
        'yielded ties. The first column ID from the platform file will be chosen '
        'for the join', Warning)

    finalMatch = matchKeys[0]
    return finalMatch


def imputeExpression(exprData, sampleList, geneID, missingGenes, gammaShape, 
    gammaLoc, scaleVals, minFluxVal, maxFluxVal):
    """ Draw samples on a set of missing genes, based on the distribution of
        expression values from the observable data (calculated in fitGamma()).
        Inherits from, and args detailed in, matchCBMGenes()
    """ 
    draws = stats.gamma.rvs(a = gammaShape, loc = gammaLoc,
        size = (len(sampleList) * len(missingGenes)))

    if any(draws < 0):
        raise ValueError('Drawing from gamma gave negative values, check expression '
            'data moments')

    if scaleVals == True:
        draws_std = (draws - np.min(draws))/(np.max(draws) - np.min(draws))
        draws = draws_std * (maxFluxVal - minFluxVal) + minFluxVal

    imputeDF = pd.DataFrame(np.reshape(draws, (len(missingGenes), len(sampleList))))
    imputeDF.columns = sampleList
    imputeDF[geneID] = missingGenes
    exprDataExtend = exprData.append(imputeDF, ignore_index = True, sort = True)

    if len(exprData) + len(missingGenes) != len(exprDataExtend):
        raise Exception('Generation of missing gene expression values failed')
    
    return exprDataExtend


def constrainModel(exprData, modelObject, geneIDMatch, geneDict, sampleName = '', 
    prepConstraints = True, lowerBound = -1000, upperBound = 1000, 
    constrainFun = 'max', **params):
    """ Constrain the metabolic model using a sample from the expression data,
        after having joined to the metabolic model genes in matchCBMGenes(). 

        Args:
            exprData: DataFrame - Expression data with metabolic model IDs.
            modelObject: Object - CobraPy or ssbio metabolic model object
            geneIDMatch: String - Column used for gene ID matching
            geneDict: Dict - Original gene IDs from the metabolic model mapped to 
                either itself or modified IDs From extractModelGeneList().
            sampleName: String - Sample ID used to constrain the model. Randomly 
                chosen by default.
            params: Dict - Parameters from .yaml
            prepConstraints: Boolean - Set bounds on the metabolic before applying
                constraints (some reactions fall outside the normal [-1000, 1000])
            lowerBound: Int/Float - Lower flux bound on all reactions in the model
            upperBound: Int/Float - Upper flux bound on all reactions in the model
            constrainFun: String - If multiple genes participate in a reaction, what
                function should be used to select flux bound? (default max)

        Returns:
            modelObject model with constrained flux bounds
    """
    if sampleName == '': #if not specified, pick random sample
        sampleName = np.random.choice([x for x in exprData.columns if 'GSM' in x])
    
    if sampleName not in exprData.columns:
        raise ValueError('Sample ID for model constraints not detected in expression '
            'data')
    
    if constrainFun not in ['max', 'min', 'mean', 'median']:
        raise ValueError('Invalid constraint function, try "max", "min", "mean", '
            'or "median"')
    
    if prepConstraints == True:
        modelObject = prepModelConstraints(modelObject, 
            lowerBound = lowerBound, upperBound = upperBound)
    
    origBoundLen = len(set([rxn.upper_bound for rxn in modelObject.reactions]))
    for rxn in modelObject.reactions:
        rxnGenes = [x.id for x in list(rxn.genes)]
        if len(rxnGenes) > 0:
            exprGenes = set()
            for x in rxnGenes:
                if x in geneDict:
                    exprGenes.add(geneDict[x])

            selectData = exprData[sampleName][exprData[geneIDMatch].isin(exprGenes)]
            
            if len(selectData) > 0:

                if constrainFun == 'max':
                    bound = np.max(selectData)
                if constrainFun == 'min':
                    bound = np.min(selectData)
                if constrainFun == 'median':
                    bound = np.median(selectData)
                if constrainFun == 'mean':
                    bound = np.mean(selectData)
                
                if modelObject.reactions.get_by_id(rxn.id).lower_bound == lowerBound:
                    modelObject.reactions.get_by_id(rxn.id).lower_bound = -bound
                if modelObject.reactions.get_by_id(rxn.id).upper_bound == upperBound:
                    modelObject.reactions.get_by_id(rxn.id).upper_bound = bound
    
    newBoundLen = len(set([rxn.upper_bound for rxn in modelObject.reactions]))
    if newBoundLen < origBoundLen:
        raise Exception('Constraints not successfully applied to the metabolic model, '
            'did you try calling resetModelBounds() first?')

    return modelObject


def prepModelConstraints(modelObject, lowerBound, upperBound):
    """ Some metabolic models display unexpected bounds (e.g. [0, 99999]). While
        there may be valid reasons the authors of these models created bounds
        which do not form to the typical [0,1000] or [-1000, 1000], I do not know
        why, and thus would rather have consistent settings. New functionality
        is being added to multiply a preset bound which is not 0 or 1000 by
        the ratio of the specific lowerBound and upperBoundn to the model maximum.
        For instance, if the max original bound is 1000, and a flux is set to a
        max of 500, if the user specifies a range of 10000, then the flux will be
        adjusted to 5000 """

    origBoundLen = len(set([rxn.upper_bound for rxn in modelObject.reactions]))
    origMax = max([rxn.upper_bound for rxn in modelObject.reactions])
    maxRatio = abs(upperBound / origMax)
    origMin = min([rxn.lower_bound for rxn in modelObject.reactions])
    minRatio = abs(lowerBound / origMin)

    counter = 0
    for rxn in modelObject.reactions:
        
        origLower = modelObject.reactions.get_by_id(rxn.id).lower_bound
        origUpper = modelObject.reactions.get_by_id(rxn.id).upper_bound
        
        if origLower < lowerBound:
            modelObject.reactions.get_by_id(rxn.id).lower_bound = lowerBound
        
        if origLower > lowerBound and origLower < 0: #If the range needs to be extended on a negative flux
            origLB = modelObject.reactions.get_by_id(rxn.id).lower_bound
            modelObject.reactions.get_by_id(rxn.id).lower_bound = max((origLB * minRatio, lowerBound)) 
        
        if origUpper > upperBound:
            modelObject.reactions.get_by_id(rxn.id).upper_bound = upperBound
            counter += 1
        
        if origUpper < upperBound and origUpper > 0: #If the range needs to be extended on a positive flux
            origUB = modelObject.reactions.get_by_id(rxn.id).upper_bound
            modelObject.reactions.get_by_id(rxn.id).upper_bound = min((origUB * maxRatio, upperBound))
        
        if origLower > origUpper and origLower > 0: #do a reset or else models may fail
            modelObject.reactions.get_by_id(rxn.id).upper_bound = upperBound
            if origLower > upperBound:
                raise ValueError('Unexpectedly high lower flux bound, check model')
        
        if origLower > origUpper and origLower < 0: #do a reset or else models may fail
            modelObject.reactions.get_by_id(rxn.id).lower_bound = lowerBound
            if origUpper < lowerBound:
                raise ValueError('Unexpectedly lower upper flux bound, check model')

    newBoundLen = len(set([rxn.upper_bound for rxn in modelObject.reactions]))

    if counter > 0:
        if origBoundLen < newBoundLen: #If successful, there should be fewer or equal values (avoid setting all to 0)
            raise Exception('Metabolic model dictionary of original flux bounds '
                'unsuccessfully built')
    
    if max([rxn.upper_bound for rxn in modelObject.reactions]) != upperBound:
        raise Exception('Metabolic model dictionary of original flux bounds '
                'unsuccessfully built')

    return modelObject


def prepEPDict(modelObject, sampleID, modelName, parseIDs = ['Title', 
    'Source name', 'Characteristics', 'Sample type', 'Organism'], 
    storeEPDir = 'GSE', protocolDir = 'params', protocol = 'GEO_GeneExprProcess.md', 
    nDevices = 2, specifyDevice = None, **params):
    """ Critical function which connects the GEO query and expression data
        processing with the MetabolicEP fitting. The format of the dict created
        here will be expected on the "receiving end" of metabolicEP. Essential
        information for metabolicEP execution will be stored under the 'Data' key,
        and params, conditions, etc will be stored under 'MetaData'. The 'Data'
        key contains the essential lb and ub information, but also reaction IDs.
        This is not necessarily, and by default will be removed anyways after 
        fitting EP, but could be useful for troubleshooting later since its cheap.
        Note that comprehensive reaction IDs, and graph stats such as node degree,
        cluster coefficient, etc, are currently housed under each metabolic models
        "graphStats.json" file in MetabolicEP/ModelRepository.
        
        Args:
            modelObject: Object - CobraPy or ssbio metabolic model object
            sampleID: Str - GEO sample identifier ('GSMxxx', e.g.)
            modelName: Str - Name of metabolic model, from reference dict
            params: Dict - Parameters from .yaml
            parseIDs: List - List of IDs to parse from sample GEO URL, which is
                performed in extractGEOSampleParams().
            storeEPDir: Str - Identifier of how to store sampleID's metabolicEP
                results (if it is to be run, which will be done outside this 
                function). Default is 'GSE', which will indicate that all samples
                under the same GSE study will be stored together. Currently, 
                storeEPDir must be contained in the dictionary fed to the 
                metabolicEP receiver function
            protocolDir: Str - Directory of GEOWrangle protocol description
            protocol: Str - Markdown file with GEOWrangle protocol description
            removeFile: Boolean - Remove series_matrix file
            nDevices: Int - Number of devices available for metabolicEP. Device
                placement will be random choice
            specifyDevice: Int - Override nDevices and force device placement
            
        Return:
            epInfoDict: Dict - Dictionary containing pertinent information to 
                pass to MetabolicEP.
    """
    if protocol not in os.listdir(protocolDir):
        raise ImportError('{0} protocol file not found in {1}'.format(protocol, 
            protocolDir))

    epInfoDict = dict() 
    epInfoDict['Data'] = dict()
    epInfoDict['MetaData'] = dict()
    for rxn in range(len(modelObject.reactions)):
        rxnID = 'v' + str(rxn)
        epInfoDict['Data'][rxnID] = dict()
        epInfoDict['Data'][rxnID]['Name'] = modelObject.reactions[rxn].id
        epInfoDict['Data'][rxnID]['lb'] = modelObject.reactions[rxn].lower_bound 
        epInfoDict['Data'][rxnID]['ub'] = modelObject.reactions[rxn].upper_bound 
    
    conditionDict = extractGEOSampleParams(sampleID, parseIDs)
    if storeEPDir not in conditionDict:
        raise Exception('{0} was not pulled from extractGEOSampleParams(): '
            'metabolicEP functions will not know where to store the '
            'results'.format(storeEPDir))

    if specifyDevice is None:
        Device = random.choice(range(0, nDevices))
    else:
        if specifyDevice not in list(range(0, nDevices)):
            raise ValueError('Specified device doesnt exist: try an integer in '
                '{0}'.format(list(range(0, nDevices))))
        else:
            Device = specifyDevice
    tag = sampleID
    if re.search(r'\ ', tag):
        warnings.warn('Tag ID contains spaces, will remove for file write safety',
             Warning)
        tag = re.sub(r'\ ', '', tag)
    
    epInfoDict['MetaData'].update(conditionDict)
    epInfoDict['MetaData'].update({'Store': storeEPDir, 'modelName': modelName,
        'Device': Device, 'tag': tag, 'Protocol': '{0}/{1}'.format(protocolDir, 
            protocol)})

    return epInfoDict


def extractGEOSampleParams(sampleID, parseIDs = ['Title', 'Source name', 
    'Characteristics', 'Sample type', 'Organism']):
    """ Extract information from a GEO sample ID ('GSMxxx'). Given the largely
        fixed structure of GEO URLs, fixed parsing criteria is hard-coded here, 
        including the sub-parsing of the Characteristics labels (hopefully GEO
        page formats wont change!). A default dict of conditions is included within
        the function, in case it is not applied externally. If additional
        information is to be added later, it either can be added as a parsing
        step, and hard-coded within this function, or it can be added later on,
        in the MetabolicEP steps, to store the information with the storeEP json.
        This function does not currently parse experimental details which would
        indicate a knockout/knockdown setting (likely to be specified under
        "Characteristics"), given the inconsistency in reporting.

        Args:
            sampleID: Str - GEO sample identifier ('GSMxxx', e.g.)
            parseIDs: List - Fields on the GEO url to parse and add to the dict
        Returns:
            expParams: Dict - Experimental parameters for the GEO sample
    
    #TODO: ADD CELL PARAM
    """
    baseParams = dict({'GSE': '', 'Title': '', 'Source name': '', 
        'Characteristics': '', 'Organism': '', 'Tissue': '', 'Sample type': '', 
        'KO': None, 'Expression': False, 'Thermodynamics': False, 
        'Variants': False, 'Protein': False, 'Metabolite': False})

    url = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={0}'.format(sampleID)
    urlGet = requests.get(url)
    urlGetText = urlGet.text 

    if 'GEO Error' in urlGetText:
        raise ValueError('GEO Error raised during the requests get(): is {0} a '
            'valid GEO sample?'.format(sampleID))

    extraIDs = [x for x in parseIDs if x not in baseParams]
    if len(extraIDs) > 0:
        warnings.warn('{0} is an unexpected experimental parameter(s), but an '
            'attempt will be made to parse based on other known paramter '
            'fields'.format(extraIDs), Warning)

    cleanText = re.sub(r'<.*?>|\\n', ' ', urlGetText) #original      
    GSE = re.findall(r'Series \(\d+\) \n     (.*)  \n', cleanText)[0]
    baseParams['GSE'] = GSE

    for i in parseIDs:
        parsed = re.findall('{0} \n (.*) \n'.format(i), cleanText)
        if len(parsed) > 0:
            cleanParse = re.sub(r'^\ (.*)', '\\1', parsed[0])
            cleanParse = re.sub(r'(.*)\ $', '\\1', cleanParse)
            if 'RNA' in cleanParse:
                baseParams['Expression'] = True
            if i == 'Characteristics' and ':' in cleanParse:
                parseOrig = re.findall(r'Characteristics<\/td>\n.*justify\">(.*)<br><\/td>\n', 
                    urlGetText)
                splits = re.split(r'<br>', parseOrig[0])
                characterDict = dict()
                for split in splits:
                    ID = re.findall(r'^(.*)\:', split)[0]
                    descript = re.findall(r'\:(.*)$', split)[0]
                    characterDict[ID] = descript
                baseParams[i] = characterDict
            else:
                baseParams[i] = cleanParse
        if len(parsed) == 0:
            print('{0} could not be parsed from sample {1} url text'.format(i, 
                sampleID))
    
    return baseParams


def executeGEOtoEP(epDirectory, epDict, handleScript = 'GEOtoEPFit.py',
    pyVersion = '3.9', defaultYAML = True, removeDict = True, 
    customResultDirTag = ''):
    """ Execute commands from the metabolicEP functions to fit EP on the GEO
        expression data. The epDict is moved to the  to the local directory for
        metabolicEP, followed by processing the epDict to create the inputs and
        parameters to MetabolicEP itself. To avoid clutter, as the conditions of
        the experiment are expected to be stored with the EP results, the epDict
        itself will finally be removed from the metabolicEP directory. Assume
        epDirectory is outside GEOWrangle, and one step away on a relative path.

        Args:
            epDirectory: Str - Location of metabolicEP functions and storage
                directories
            epDict: Dict - Constrants and conditions information, from prepEPDict()
            handleScript: Str - Python script on the "receiving end" to execute
                MetabolicEP
            pyVersion: Str - Python version to call from shell command
            defaultYAML: Boolean - Use default parameters' .yaml file in MetabolicEP.
                Currently should only be True, as other parameters, or user-defined
                parameters, are not a part of the handleScript functionality (
                    future direction).
            removeDict: Boolean - Remove the epDict from the epDirectory after
                calling subprocess
            customResultDirTag: Str - If results are to be placed under a non-default
                result directory under epDirectory (currently set to 'resultDir'
                as a default param in handleScript). Must be a tag attached to
                'resultDir', not the WHOLE folder name (avoid slashes as args).
        
        Returns:
            Nothing. However, metabolicEP should be run on the contents of
            epDict and saved within the epDirectory
    """
    tempTag = epDict['MetaData']['tag'] + '_epTemp.json'
    fileName = '../{0}/{1}'.format(epDirectory, tempTag)
    with open(fileName, 'w') as fStore:
        json.dump(epDict, fStore, indent = 4)
    
    if handleScript not in os.listdir('../{0}'.format(epDirectory)):
        raise FileNotFoundError('GEO to MetabolicEP execution script ({0}) not '
            'found in ../{1}'.format(handleScript, epDirectory))

    if customResultDirTag == '':
        executeScript = 'python{0} ../{1}/{2} {1} {3} {4}'.format(pyVersion, 
            epDirectory, handleScript, tempTag, defaultYAML)
    else:
        executeScript = 'python{0} ../{1}/{2} {1} {3} {4} {5}'.format(pyVersion, 
            epDirectory, handleScript, tempTag, defaultYAML, customResultDirTag)

    subprocess.call(executeScript, shell = True, executable = '/bin/bash')

    if removeDict == True:
        os.remove(fileName)


def resetModelBounds(modelObject, boundDict):
    """ Reset the flux bounds to original values, using a dictionary created in
        prepModelConstraints(). Resetting values saves time over re-reading the
        model object between each sample. """

    origBoundLen = len(set([rxn.upper_bound for rxn in modelObject.reactions]))
    counter = 0
    for rxn in modelObject.reactions:
        
        if rxn.id in boundDict:
            origLower = boundDict[rxn.id]['Lower bound']
            origUpper = boundDict[rxn.id]['Upper bound']

            modelObject.reactions.get_by_id(rxn.id).lower_bound = origLower

            if modelObject.reactions.get_by_id(rxn.id).upper_bound != origUpper:
                modelObject.reactions.get_by_id(rxn.id).upper_bound = origUpper
                counter += 1

        else:
            warnings.warn('Warning: while attempting to reset the flux bounds, '
                'the reaction ID was not found in the flux bound dictionary', Warning)
    
    newBoundLen = len(set([rxn.upper_bound for rxn in modelObject.reactions]))

    if counter > 0:
        if origBoundLen <= newBoundLen:
            raise Exception('Resetting of metabolic model flux bounds unsuccessful')

    return modelObject


def queryListGEOQuery(queryTextFile, organism, geoFilter = 'GSE', studyType = ''):
    """ Return the set of GSEs from a GEO query on organism, and possibly an added
        partial 'DataSet Type' (e.g. 'expression', instead of 'expression profiling
        'by array'). Full query result can be written to file from the url and
        read in and parsed for GSE list (may be horrifically slow though, kb/s).
        Dependent formatting is from exporting a GEO query as file (under the
        'Send to:' dropdown)

        Args:
            queryTextFile: Str - .txt file exported from a GEO query.
            organism: Str - Species, assumed two words with space separation (e.g. 
                Homo sapiens).
            studyType: Str - Partial string match for a type of data, e.g. 
                'expression'. Note that 'expression' is not a valid 'DataSet Type',
                so the 'Type' field will need to be parsed.
            filter: Str - Assumed we want GEO series ('GSE'), however one could
                also extract all samples ('GSM'), platforms ('GPL'), or DataSets
                ('GDS') 
        Returns:
            queryList: List - List of GSE (or GSM, GPL, etc) for a given GEO query
    """
    with open(queryTextFile, 'r') as fin:
        content = fin.read()
    geoSplit = re.split('\\n\\n', content)
    geoSplit = [x for x in geoSplit if re.findall('\\nOrganism\:\\t(.*)\\n', 
        x)[0] == organism]
    if studyType != '':
        geoSplit = [x for x in geoSplit if studyType.lower() in re.findall('\\nType\:\\t\\t(.*)\\n', 
            x)[0].lower()]
    queryList = list(set([re.findall('Accession\: ({0}\d+)\\t'.format(geoFilter), 
        x)[0] for x in geoSplit]))
    
    return queryList
