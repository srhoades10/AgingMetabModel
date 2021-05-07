import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing, decomposition
import warnings


def rmMissingness(exprData, obsCutoff, probeCutoff):
    """ Remove high missingness from the expression data, where we define unique
        cutoffs for both observations and probes. As an example, a cutoff of 0.
        would mean that every sample and prob will be removed, while 1. keeps all
        values, and 0.75 would remove samples or probes which have values for at
        least 25% of instances.

        Args:
            exprData: Gene expression data
            obsCutoff: Fraction of valid probe values required to keep an observation.
                    Value should be between 0 and 1
            probeCutoff: Fraction of observations required to keep a probe.
                    Value should be between 0 and 1
        
        Returns:
            exprData with probes removed with high missingness
    """
    if probeCutoff < 0. or probeCutoff > 1. or obsCutoff < 0. or obsCutoff > 1.:
        raise ValueError('Cutoffs should be fractions between 0 and 1')
    
    exprData = exprData[exprData.columns[(exprData.isnull().sum(axis = 0) < 
        obsCutoff * len(exprData))]]
    exprData = exprData[(exprData.isnull().sum(axis = 1) < 
        probeCutoff * len(exprData.columns))]
    exprData = exprData.reset_index(drop = True)
    return exprData


def imputeMissing(exprData, imputeStrategy):
    """ Impute missing values and put in unit variance, a la sklearn. Given the
        input data frames are from GEO, we assume the data has probes in rows 
        and samples in columns. 

        *** Point of concern! I don't love basic "median" imputation, we can
            do better, but this is the most basic and fast. And missing values
            in microarray data is very rare. *** 
            
        Args:
            exprData: Gene expression dataframe, with the ID column assumed in
                the first column (if this is not true, error would inherit from 
                filterObs).
            imputeStrategy: How should we impute? Median is likely better than
                mean in skewed data
        
        Returns:
            exprData with missing values imputed
    """
    if any(exprData.isna().sum() > 0):
        warnings.warn('Expression data appears to have missing values. Imputation '
            'will be done. Default imputation strategy is median.', Warning)
        colNames = exprData.columns
        dataOnly = exprData[exprData.columns[1:]]
        impute = preprocessing.Imputer(missing_values = 'NaN', 
            strategy = imputeStrategy, axis = 1).fit(dataOnly)
        dataImpute = pd.DataFrame(impute.transform(dataOnly))
        if dataImpute.shape != dataOnly.shape:
            raise Exception('Warning: Data shape changed after imputation, which '
                'likely means a probe had all missing values, and no filtering '
                'was applied in rmMissingness()')
    
        if any(dataImpute.isna().sum() > 0):
            raise Exception('Imputation failed, nans still present')

        exprData = pd.concat([exprData.ID_REF, pd.DataFrame(dataImpute)], 
            axis = 1)
        exprData.columns = colNames
    
    return exprData


def logTransform(exprData, minExprValue, skewLimit, suppressWarning = False):
    """ Log transform will be applied if the max expression value is greater
        than 1000, and if the log-transform doesn't change the data distribution
        to a "normal" one (skew < skewLimit). Note that the original data was already
        filtered in GEOWrangle with skewLimit. Data will also be adjusted to minVal
        
        Args:
            exprData: Gene expression dataframe, with the ID column assumed in the
                the first column (if this is not true, error would inherit from 
                filterObs).
            minExprValue: Float - Value to add to the quantitative data in exprData, 
                on top the minimum value of the exprData.
            skewLimit: Float - Threshold to apply log transformation
            suppressWarning: Boolean: suppress warnings for cleaner unittest prints
        
        Returns:
            exprData with log-transformation, if necessary 
    """
    colNames = exprData.columns
    dataOnly = exprData[exprData.columns[1:]]
    dataAdj = adjMinValue(dataOnly, minExprValue) #for log2

    vals = np.ravel(dataAdj)
    vals = vals[~np.isnan(vals)]
    if any(vals > 1000.):
        if stats.skew(np.log2(vals)) < skewLimit:
            if suppressWarning == False:
                warnings.warn('Log-transform appears to have dropped skewness '
                    'below {0}, original data will be kept as-is'.format(skewLimit),
                     Warning)
            exprData = pd.concat([exprData.ID_REF, dataAdj], axis = 1)
            return exprData
        else:
            dataAdj = np.log2(dataAdj)
            exprData = pd.concat([exprData.ID_REF, pd.DataFrame(dataAdj)], 
                axis = 1)
            exprData.columns = colNames
            return exprData
    else:
        exprData = pd.concat([exprData.ID_REF, dataAdj], axis = 1)
        return exprData


def adjMinValue(exprDataOnly, minExprValue):
    """ Adjust all quantitative values of the expression data to a minimum of 0, 
        plus some additional minValue if you want (e.g. if you need to take logs 
        later). This function is currently (9/17/2018) run inside logTransform()
        only if values are high and assumed not scaled (in which case log2s are
        taken and the minValue is set to 1.0).
        
        Args:
            exprDataOnly: Gene expression dataframe, with the ID column excluded.
                Inherited from preprocessExpr.
            minValue: Value to add to the quantitative data in exprData, on top
                 the minimum value of the exprData. Inherited from preprocessExpr.
        
        Returns:
            Adjusted exprData with all quantitative values now positive
    """
    if minExprValue < 0:
        raise ValueError('Minimum value must be >= 0')

    Mins = abs(min(exprDataOnly.min())) + minExprValue
    exprDataAdj = exprDataOnly.add(Mins)
    return exprDataAdj


def rmPCAOutliers(exprData, hotellings):
    """ Remove observations from the expression data based on PCA. The 
        percentileCutoff is like the Hotellings for PCA scores. 95% is 2 SDs.
        Currently only the first 2 components are used for hotellings, while a
        third may be useful, I am not confident the 3D hotelling function is 
        working quite right. More than 3D is probably not worth it.
    
        Args:
            exprData: Gene expression data
            hotellings: Cutoff for outlier detection (i.e. Hotellings %). As a
                percent, it should be between 0 and 100
        
        Returns:
            exprData with outlier observations removed
    """
    if hotellings > 100 or hotellings < 0:
        raise ValueError('Percentiles must be between 0 and 100')

    if hotellings <= 1 and hotellings >= 0:
        warnings.warn('Warning: percentileCutoff is between 0 and 1, did you mean '
            'to enter such a low percentage? You can enter any number between 0 '
            'and 100, with 95 representing 2 SDs', Warning)
    
    sampleNames = list(exprData.columns[1:].values)
    dataOnly = exprData[exprData.columns[1:]].T #NxM
    dataOnly.reset_index(inplace = True, drop = True) #remove colNames as indices
    dataProjections = fitPCA(dataOnly)

    radiusX, radiusY = calc2DHotellings(dataProjections, hotellings)
    dropIndex = flag2DOutliers(dataProjections, radiusX, radiusY)
    dataOnly = dataOnly.drop(dropIndex, axis = 0)

    if len(dropIndex) > 0:
        for i in sorted(dropIndex, reverse = True):
                del sampleNames[i]

    exprData = pd.concat([exprData.ID_REF, dataOnly.T], axis = 1)
    exprData.columns = ['ID_REF'] + sampleNames
    return exprData


def fitPCA(exprDataOnlyTranspose):
    """ Fit PCA to the transposed expression data (NxM) inherited from 
        rmPCAOutliers(), and returns the projections in PC space. Uses sklearn """
    
    scaler = preprocessing.StandardScaler().fit(exprDataOnlyTranspose)
    dataTransform = scaler.transform(exprDataOnlyTranspose)
    pca = decomposition.PCA()
    pcaFit = pca.fit(dataTransform)
    dataProjections = pcaFit.transform(dataTransform)
    return dataProjections


def calc2DHotellings(pcaProjections, hotellings):
    """ Calculate the radius of the hotellings circle based on the PCA scores. 
        Scores are inherited from rmPCAOutliers.

        Args:
            pcaProjections: Scores from PCA-transformed dataset
            hotellings: Cutoff for outliers. Inherits from rmPCAOutliers().
        
        Returns:
            x and y ellipse radius for hotellings confidence    
    """
    theta = np.concatenate((np.linspace(-np.pi, np.pi, 50), 
        np.linspace(np.pi, -np.pi, 50)))
    circle = np.array((np.cos(theta), np.sin(theta)))
    sigma = np.cov(np.array((pcaProjections[:, 0], 
        pcaProjections[:, 1])))
    ed = np.sqrt(stats.chi2.ppf(hotellings / 100, 2)) #2 df

    try:
        ellipse = np.transpose(circle).dot(np.linalg.cholesky(sigma) * ed)
        x, y = np.max(ellipse[: ,0]), np.max(ellipse[: ,1])
        if x < y:
           raise ValueError('t1 is less than t2, check PCA for extreme distortions '
                'in the data and that maximal variance is along 1st axis')
        elif np.isnan(x) or np.isnan(y):
            raise ValueError('t1 or t2 is nan, Hotellings calculation likely failed')
        else:
            return x, y

    except np.linalg.linalg.LinAlgError:
        raise ArithmeticError('Cholesky failed, check PCA for successful SVD and '
            'that there arent columns with 0 variance')
    except:
        raise Exception('Unknown error in hotelling ellipse calculation')


def flag2DOutliers(pcaProjections, radiusX, radiusY):
    """ Extract indices of outliers based on hotellings ellipse radii, using the
        ellipse equation
        
        Args:
            pcaProjections: Projections of obs onto PC space. Inherits from 
                rmPCAOutliers()
            radiusX: Hotellings ellipse t1. Inherited from calc2DHotellings()
            radiuxY: Hotellings ellipse t2. Inherited from calc2DHotellings()
        
        Returns:
            Row indices (observations) which lie outside Hotellings space and are
            to be excluded
    """
    outLocs = []
    for obs in range(len(pcaProjections)):
        location = (pcaProjections[obs, 0]**2 / radiusX**2 + 
            pcaProjections[obs, 1]**2 / radiusY**2)
        if location > 1.:
            outLocs.append(obs)
    return outLocs


def filterCV(exprData, absoluteCVCutoff, meanCVThreshold = False):
    """ Remove probes which exhibit a high coefficient of variation. And, for the
        lowly-expressed genes (which we don't want to remove since they can be 
        valid constraints on the metabolic model), below the dataset mean value,
        apply an additional meanCVThreshold to set those probes to their mean to
        both improve stability of MAD scaling and keep their percentile low.

        Args:
            exprData: Gene expression data
            absoluteCVCutoff: Numeric: high CV cutoff to drop from the dataset
            meanCVThreshold: Numeric: CV threshold (l.t.e absoluteCVCutoff)
                above which probes are set to their mean, if that mean is lower
                than the mean for the entire dataset. Default is to skip this.
        
        Returns:
            exprData where probes with high CV are removed or adjusted
    """
    if (not isinstance(absoluteCVCutoff, int) 
        and not isinstance(absoluteCVCutoff, float)
        or float(absoluteCVCutoff) < 0):
        raise ValueError('CV cutoff must be a numeric positive real')

    if absoluteCVCutoff <= 1 and absoluteCVCutoff >= 0:
        warnings.warn('Warning: absoluteCVCutoff is between 0 and 1, did you mean '
            'to enter such a low percentage?', Warning)

    CVs, probeMeans = calcCVs(exprData)

    if meanCVThreshold != False:
        if (not isinstance(meanCVThreshold, int) 
            and not isinstance(meanCVThreshold, float)):
            raise ValueError('Enter an integer or float for meanCVThreshold')
        elif absoluteCVCutoff < meanCVThreshold:
            raise ValueError('absoluteCVCutoff must be greater than or equal to '
                'meanCVThreshold')
        elif meanCVThreshold <= 1 and meanCVThreshold >= 0:
            warnings.warn('Warning: meanCVThreshold is between 0 and 1, did you mean '
                'to enter such a low percentage?', Warning)
            exprData = setMeanProbeVals(exprData, meanCVThreshold, probeMeans, CVs)
        else:
            exprData = setMeanProbeVals(exprData, meanCVThreshold, probeMeans, CVs)
    
    dataTrim = exprData[(CVs < absoluteCVCutoff)]
    dataTrim = dataTrim.reset_index(drop = True)

    if len(dataTrim) == 0:
        raise Exception('All probes were removed, try a different CV cutoff')

    return dataTrim


def calcCVs(exprData):
    """ Probe means and CVs from expression data. Inherit from filterCV() """
    
    exprDataOnly = exprData[exprData.columns[1:]]
    probeMeans = np.nanmean(exprDataOnly, axis = 1)
    probeSDs = np.nanstd(exprDataOnly, axis = 1)
    try:
        CVs = np.abs((probeSDs / probeMeans) * 100)
    except RuntimeWarning:
        raise ValueError('Mean value for at least one probe is exactly 0, CV '
            'cannot be calculated')
    except:
        raise ValueError('Unknown CV calculation error')
    return CVs, probeMeans


def setMeanProbeVals(exprData, meanCVThreshold, probeMeans, probeCVs):
    """ For probes which are above the CV threshold, and whose mean is below the 
        mean value for the entire expression dataset, broadcast their probe's mean
        values in order to fill into the expression data values. All Args
        inherited from filterCV(). Note this is an optional function, as the 
        decision to set a lowly-expressed probe to a mean value, as opposed to 
        removing it, is purely arbitrary. However, if the Gamma-fitting and 
        imputation is used in GEOWrangle, the lowly-expressed genes, which would
        be removed with a CV cutoff, would have a high probability of being
        drawn as a lowly-expresed gene anyways (if we're operating under skewed
        expression distributions).

        Args:
            exprDataOnly: Expression data without probe IDs
            meanCVThreshold: Numeric: CV threshold (lower than absoluteCVCutoff)
                above which probes are set to their mean, if that mean is lower
                than the mean value for the entire dataset
            probeMeans: Mean value for each probe. Generated from calcCVs().
            probeCVs: CVs by probe. Generated from calcCVs().
        
        Returns:
            probeMeans broadcast to the shape of exprData
    """
    dataOnly = exprData[exprData.columns[1:]]
    datasetMean = np.nanmean(np.ravel(dataOnly))
    if datasetMean < 0: #this should never be raised given the skewness check, but..
        raise ValueError('Mean expression value for the entire dataset is negative, '
            'check data processing steps and inspect distribution of values')
    
    thresholdMeans  = np.nanmean(dataOnly[(probeCVs > meanCVThreshold) & 
        (probeMeans < datasetMean)], axis = 1)
    if len(thresholdMeans) > 0:
        meanBroadcast = pd.DataFrame(np.broadcast_to(np.expand_dims(thresholdMeans, 1), 
            (len(thresholdMeans), len(dataOnly.columns))))
        dataOnly.loc[(probeCVs > meanCVThreshold) & 
            (probeMeans < datasetMean)] = meanBroadcast.values
        exprDataNew = pd.concat([exprData.ID_REF, dataOnly], axis = 1)
        exprDataNew.columns = exprData.columns
        return exprDataNew
    else:
        return exprData


def scaleExpression(exprData, axis = 0, coef = 2, suppressWarning = False):
    """ Adjust the expression data by the 'invented' median-based scaling approach
        defined in adjustedMedianScaling() (see 9/30/2018 lab notebook entries). 
        The assumption is to scale across samples for each probe (axis = 1), in
        order to add 'weight' to the influence of a probe's expression across
        samples, in effect combining the differential expression notion, which is
        in gene expression studies, with the Pareto Principle of high and low 
        gene expressors. If axis is set to 0, then the resulting distribution of
        expression is essentially Normalized(), which is (unfortunately) common
        in gene expression data processing.

        Apr 10, 2019 update: After classifier optimization on processing
            parameters (see OptConstrain), axis = 0 and coef = 2 were shown to 
            improve performance, and will be set as defaults here.

        Args:
            exprDataOnly: Expression data without probe IDs. Inherited from 
                scaleExpression(). Assumed to be used in an apply() call.
            axis: Scale across probes (axis = 0), essentially normalizing the
                entire dataset, or scale across samples (axis = 1), which adds
                an influence of differential expression to a probe across samples.
            coef: Numeric. Strength of scaling. Inherited from scaleExpression().
                Default value is 4, based on empirical tests on 9/30/2018.
                Lower values results in stronger scaling.
            suppressWarning: Boolean: suppress warnings for cleaner unittest 
                outputs
        
        Returns:
            Expression data with scaled values
    """
    if axis == 1 and suppressWarning == False:
        warnings.warn('Warning: You elected to scale across samples, not probes, '
            'which has been tested to improve predicted flux classification '
            'performance. See DoE results in OptConstrain for details', Warning)
    if axis != 1 and axis != 0:
        raise ValueError('Invalid axis argument, enter 0 or 1')
    
    if (not isinstance(coef, int) 
        and not isinstance(coef, float)
        or float(coef) < 0):
        raise ValueError('CV cutoff must be a numeric positive real')
    if float(coef) < 2 and suppressWarning == False:
         warnings.warn('Warning: you chose to apply a stronger differential '
            'expression scaling factor', Warning)
    if float(coef) > 2 and suppressWarning == False:
         warnings.warn('Warning: you chose to apply a weaker differential '
            'expression scaling factor', Warning)

    dataOnly = exprData[exprData.columns[1:]]
    dataOnly = dataOnly.apply(adjustMedianScaling, axis = axis, raw = True, 
        args = (coef, False, ))
    exprDataNew = pd.concat([exprData.ID_REF, dataOnly], axis = 1)
    if exprData.shape != exprDataNew.shape:
        raise Exception('New dataframe is not of the original shape, scaling '
            'failed')
    else:
        exprDataNew.columns = exprData.columns
        return exprDataNew


def adjustMedianScaling(exprDataOnly, coef = 4, checkArgsort = False):
    """ Adjust values by a modified deviation from the median. The 'strength' of
        the scaling is adjusted by a multiplication of that divide-by-median by
        a factor. This factor is a trade-off of how much we want to weigh the
        differential expression across samples. coef = 4 seemed like a reasonable
        tradeoff of factoring differential expression without spiking the variance
        in the data (see 9/30/2018 lab notebook entries). Coef = 1 is similar to
        the MAD approach, but can result in highly-inflated CV%s and skews, and
        generally caused problems in low-expressors.
        
        Args:
            exprDataOnly: Expression data without probe IDs. Inherited from 
                scaleExpression(). Assumed to be used in an apply() call.
            coef: Numeric. Strength of scaling. Inherited from scaleExpression().
                If run indepedently, default is set to 4 based on empirical
                evidence on 9/30/2018.
            checkArgsort: Check that value ordering doesn't change pre and post
                scaling. Important during dev of the new approach, but np.argsort()
                chews up time, thus this is reserved for unittests. Note this
                is NOT inherited from scaleExpression, as it'll only be used in 
                unit-testing within this function.
        
        Returns:
            returnToScale: Scaled expression data
    """
    allVals = np.ravel(exprDataOnly)
    if np.min(allVals) < 0:
        raise ValueError('Negative values shouldnt exist! Scaling failed')

    medianVal = np.median(exprDataOnly)
    scaleFactor = medianVal * coef
    adjustValues = exprDataOnly * (exprDataOnly - medianVal) / scaleFactor
    returnToScale = adjustValues + exprDataOnly

    if checkArgsort == True:
        originalOrder = np.argsort(exprDataOnly)
        newOrder = np.argsort(returnToScale)
        if not all(originalOrder == newOrder):
            raise Exception('Ordering of values changed during scaling')

    return returnToScale


def scaleCBM(exprData, dropCol = 'ID_REF', minFluxVal = 0, maxFluxVal = 1000):
    """ Scale the expression values to interval [0, maxVal], suitable for CBM.
        This approach does not work with negative values, however it is assumed
        that logTransform() was already applied to the data. Functions taken from
        sklearn's MinMaxScaler.
    
        Args:
            exprData: Gene expression data
            dropCol: Str - Assuming expression data only has one ID column, this
                column is removed before scaling, followed by rejoining at the end.
                Default is ID_REF, but could also apply for expression datasets
                which have an alternative gene or probe ID.
            minFluxVal: Desired minimum (maybe 0 is not good in CBM?)
            maxFluxVal: Desired maximum value on the scaler
        Returns:
            exprData in the interval [0, maxVal]
     """
    if maxFluxVal != 1000:
        warnings.warn('Warning: you chose a different scale then the [0, 1000] '
            'interval common in constraint-based modeling approaches', Warning)
    
    if dropCol not in exprData.columns:
        raise Exception('Expression data probe IDs not found: invalid data format')

    colNames = exprData.columns
    dataOnly = exprData.drop(dropCol, axis = 1)
    vals = np.ravel(dataOnly)
    vals_std = (vals - np.min(vals))/(np.max(vals) - np.min(vals))
    scaleVals = vals_std * (maxFluxVal - minFluxVal) + minFluxVal
    convertData = np.reshape(scaleVals, (dataOnly.shape[0], dataOnly.shape[1]))
    exprData = pd.concat([exprData[dropCol], pd.DataFrame(convertData)], axis = 1)
    exprData.columns = colNames
    return exprData


def processData(expressionDF, obsCutoff = 0.25, probeCutoff = 0.25, 
    imputeStrategy = 'median', minExprValue = 1.0, skewLimit = 0.25, 
    hotellings = 99.7, absoluteCVCutoff = 200, meanCVThreshold = False, axis = 1, 
    coef = 4, dropCol = 'ID_REF', minFluxVal = 0, maxFluxVal = 1000, **params): 
    """ Collective data processing steps from an input expression dataset from GEO.
        Rather than copying the descriptions of all these arguments, the
        descriptions can be found in the individual function's Docstrings.

        Args:
            expressionDF: Expression data from GEO, read with experimentFilesReader()
                in setup_GEOWrangle.py
        
        Returns:
            expressionDF with probes and samples with high missingness removed, 
                NAs imputed, log-transformed (if necessary), outlier-removed,
                CV-filtered, and scaled 
    """
    if expressionDF.columns[0] != 'ID_REF':
        raise Exception('Transcript IDs not in first column: check data format')

    expressionDF = rmMissingness(expressionDF, obsCutoff = obsCutoff, 
        probeCutoff = probeCutoff)
    expressionDF = imputeMissing(expressionDF, imputeStrategy = imputeStrategy)
    expressionDF = logTransform(expressionDF, minExprValue = minExprValue, 
        skewLimit = skewLimit)
    expressionDF = rmPCAOutliers(expressionDF, hotellings = hotellings)
    expressionDF = filterCV(expressionDF, absoluteCVCutoff = absoluteCVCutoff, 
        meanCVThreshold = meanCVThreshold)
    expressionDF = scaleExpression(expressionDF, axis = axis, coef = coef)
    expressionDF = scaleCBM(expressionDF, dropCol = dropCol, 
        minFluxVal = minFluxVal, maxFluxVal = maxFluxVal)
    
    return expressionDF


def createPosToyPCAData(outliers = True):
    """ Create positive control toy PCA data to check for correct hotelling t1 and
        t2 calculation and outlier removal. Returns PCA projections and data """
    np.random.seed(1)
    data = np.array(np.random.normal(0, 1, 500)).reshape(100, 5)
    if outliers == True:
        outliers = np.array(np.random.uniform(5, 10, 25)).reshape(5, 5)
        dataAdd = np.vstack((data, outliers))
    else:
        dataAdd = data
    pca = decomposition.PCA(n_components = 3)
    scaler = preprocessing.StandardScaler().fit(dataAdd)
    dataAddTransform = scaler.transform(dataAdd)
    pcaFit = pca.fit(dataAddTransform)
    dataProjections = pcaFit.transform(dataAddTransform)
    return dataAdd, dataProjections

def createNegToyPCAData():
    """ Create neagtive control toy PCA data to check for pca errors """
    data = np.transpose(np.repeat([1., 2., 3., 4., 5.], 100).reshape(100, 5))
    pca = decomposition.PCA(n_components = 3)
    scaler = preprocessing.StandardScaler().fit(data)
    dataTransform = scaler.transform(data)
    pcaFit = pca.fit(dataTransform)
    dataProjections = pcaFit.transform(dataTransform)
    return data, dataProjections


def exprToPercentile(exprData):
    """ Convert the expression dataset to percentiles between 0-1000 for CBM constraints.
        Currently not in use, as the hypothesis is that biological systems are NOT
        rank-based.
    
        Args:
            exprData: Gene expression dataset, after data processing steps (at least
                requires rmMissingness(), imputeMissing(), and logTransform()).
        Returns:
            exprData with rank-based values between 0 and 1000
    """
    colNames = exprData.columns
    exprDataOnly = exprData[exprData.columns[1:]]
    vals = np.ravel(exprDataOnly)
    percents = stats.rankdata(vals)*100/len(vals)
    percents = np.multiply(percents, 10)
    convertData = pd.DataFrame(np.reshape(percents, (exprDataOnly.shape[0], 
        exprDataOnly.shape[1])))
    exprData = pd.concat([exprData.ID_REF, convertData], axis = 1)
    exprData.columns = colNames
    return exprData