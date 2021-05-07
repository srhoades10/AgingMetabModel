import numpy as np 
import cupy as cp
import re, random, math, warnings, os, json
from scipy.special import erf as sciErf
from random import shuffle

dtype = np.float64
np.seterr(divide = 'ignore') #Zero-divide should get adjusted with minVar/maxVar
c1 = 1. / np.sqrt(2.0) #Avoid calling np.sqrt millions of times
c2 = 1. / np.sqrt(2. * np.pi)


def prepMetabolicModel(modelName, baseDict = 'modelDict.json', 
    modelPath = 'ModelRepository', toy = False, **params):
    """ Prep inputs to MetabolicEP from metabolic model file 

        Args:
            modelName: Str - Model name
            params: Dict - Parameter file, from .yaml
            baseDict: Str - Pointers to S, lb, and ub of metabolic models
            modelPath: Str - Folder containing lbs, ubs, S for each model
            toy: Boolean - Use a toy model (for testing purposes)
        Returns:
            S: Array - Stoichiometric matrix
            lbs: Array - Lower bounds
            ubs: Array - Upper bounds
    """
    if toy == True:
        S = np.array([[-1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0,-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0,-1.0, 0.0, 0.0, 0.0, 1.0],
                        [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0, 0.0,-1.0, 0.0]]).astype(np.float32)
        nRxns, varShape = S.shape[1], [S.shape[1], 1]
        lbs = np.zeros(nRxns, dtype).reshape(varShape)
        ubs = np.ones(nRxns, dtype).reshape(varShape)
        return S, lbs, ubs

    else:
        with open('{0}/{1}'.format(modelPath, baseDict), 'r') as fIn:
            modelDict = json.load(fIn)
            
        if modelName not in modelDict:
            raise ValueError('Model name not found in set of possible models, '
                'use one of {0}'.format(modelDict.keys()))

        S = np.load(modelDict[modelName]['S']).astype(dtype)
        varShape = [S.shape[1], 1]
        lbs = np.load(modelDict[modelName]['lbs']).astype(dtype).reshape(varShape)
        ubs = np.load(modelDict[modelName]['ubs']).astype(dtype).reshape(varShape)
        
        if len(lbs) != len(ubs) or len(lbs) != S.shape[1]:
            raise ImportError('Stoichiometric matrix shape improper, reaction '
                'length not equivalent to flux bound length')
        
        return S, lbs, ubs


def checkArgs_prepEP(S, priorLBs, priorUBs, modelName,
    mseThreshold = 1e-06, Beta = 1e07, minVar = 1e-50, maxVar = 1e50, damp = 0.7,
    maxIter = 2000, locInit = 0., scaleInit = 0.5, baseDict = 'modelDict.json', 
    modelPath = 'ModelRepository', resultDir = 'resultStore', 
    storeEPRoot = 'storeEP.json', priorData = '', subResultFolder = '', **params):
    """ Check arguments before running EP. Return objects to be used in 
        MetabolicEP, and optionally, use priorData housed in storeEP.json to 
        update the objects with 'priors' (if using the base, unconstrained case, 
        then priorData can be set to 'Base'). The purpose of separate storeEPRoot
        and priorData arguments is to allow an easier specification of where to
        read prior data ("Base", or "GSMxxx", instead of knowing the whole file)

        Args:
            S: Array - Stoichiometric metabolic matrix
            priorLBs: Array - Lower flux bounds 
            priorUBs: Array - Upper flux bounds
            modelName: Str - Name of metabolic model (see modelDict)
            params: Dict - Parameter settings from .yaml
            mseThreshold: Float - MSE threshold for convergence 
            Beta: Float - temperature parameter
            minVar: Float - Minimum allowable variance
            maxVar: Float - Maximum allowable variance 
            damp: Float - "Strength" of mean/var updating (1 is no adjustment, 0 is
                complete replacement of posterior value)
            maxIter: Int - Number of iterations to run until break (to avoid runaway)
            locInit: Float/Array - Initialized value of flux locations (could be 
                set with an array of priors, based on experimental data)
            scaleInit: Float/Array - Initialized value of flux scales (could be 
                set with an array of priors, based on experimental data)
            resultDir: Str - Directory of stored EP results
            storeEPRoot: Str - File for prior result storage
            priorData: Str - Experimental data stored under the modelName in 
                the resultDir dictionary. Pieced together with storeEPRoot.
            subResultFolder: Str - Additional directory under resultDir, e.g.
                for storing GSE-specific EP.

         Returns:
            b: Array - Steady-state vector (presumably 0s) of metabolites
            unc_rvLoc: Array - Mean of the posterior distribution (unconstrained)
            unc_rvVar: Array - Variance of the posterior distribution (unconstrained)
            priorLoc: Array - Means of the approximated priors
            priorVar: Array - Variances of the approximated priors
            conLoc: Array - Average of the truncated Gaussians of the approximation
            conVar: Array - Variances of the truncated Gaussians of the approximation
            coVar: Array - Final covariance matrix
            factor: Float - Scaling factor
            lbs: Array - Scaled lower bounds 
            ubs: Array - Scaled upper bounds
            D: Array - nRxn x nRxn matrix, with non-zero diagonals (from variances) 
            KK: Array - Reaction adjacency matrix scaled by Beta
            KB: Array - Reaction vector scaled by Beta
    """
    if not all(isinstance(x, float) for x in [mseThreshold, Beta, minVar, maxVar, 
        damp, locInit, scaleInit]):
        raise ValueError('MSE, Beta, minVar, maxVar, damp, locInit, scaleInit must '
            'all be floats')
    if not isinstance(maxIter, int):
         raise ValueError('Max iterations must be integer')
    if damp < 0. or damp > 1.:
        raise ValueError('Posterior update dampening must be between 0 and 1')
    if scaleInit <= 0.:
        raise ValueError('Invalid scale initialization, must be positive float')
    if any(priorLBs > priorUBs):
        raise ValueError('Lower bounds greater than upper bounds detected')

    modelFolder = re.sub(r'\ |\.', '', modelName)
    if priorData != '':
        if subResultFolder != '' and len(os.listdir('{0}/{1}/{2}'.format(resultDir,
            modelFolder, subResultFolder))) == 0:
            warnings.warn('Specified prior experiment ({0}) not found, will resort '
                'to using Base priors'.format(subResultFolder), Warning)
            storeEPRoot = 'Base' + '_' + storeEPRoot #switch to, base
        elif subResultFolder != '' and priorData == 'Base': #Base is stored above and takes priority
            storeEPRoot = priorData + '_' + storeEPRoot            
            modelFolder = modelFolder
        else:
            storeEPRoot = priorData + '_' + storeEPRoot            
            modelFolder = modelFolder + '/' + subResultFolder
        if storeEPRoot not in os.listdir('{0}/{1}'.format(resultDir, modelFolder)):
            raise FileExistsError('Prior data file {0} not found in '
                '{1}/{2}'.format(storeEPRoot, resultDir, modelFolder))
        else:
            print('Attempting to use {0} for MetabolicEP priors'.format(storeEPRoot))

    with open('{0}/{1}'.format(modelPath, baseDict), 'r') as fIn:
        modelDict = json.load(fIn)
    if modelName not in modelDict:
        raise ValueError('Model name not found in set of possible models, '
                'use one of {0}'.format(modelDict.keys()))

    if Beta > 1e09:
        warnings.warn('Specified Beta is very high, unknown behavior', Warning)
    if minVar < 1e-50 or maxVar > 1e50:
        warnings.warn('Unknown behavior beyond minVar of 1e-50 or maxVar of 1e50')
    
    (b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor,
        lbs, ubs) = prepInputs(S, priorLBs, priorUBs, locInit, scaleInit, 
            modelName, resultDir, storeEPRoot, priorData, modelFolder)

    D, KK, KB = prepMatricies(S, b, priorVar, Beta)

    return (b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor, 
        lbs, ubs, D, KK, KB)


def prepInputs(S, lbs, ubs, locInit, scaleInit, modelName, resultDir, storeEPRoot,
     priorData, modelFolder):
    """ Prep inputs to MetabolicEP, including scaling flux bounds. Inherits from
        runEP. For each of the 3 loc and 3 var objects, checks will be made that
        each has an entry in the priorDataFile, if not, then the variable will
        be set to the loc or scaleInit. """
    
    nMetabs, nRxns = S.shape[0], S.shape[1]
    b = np.reshape(np.zeros(nMetabs, dtype), (nMetabs, 1))

    factor = np.max([np.max(np.abs(lbs)), np.max(np.abs(ubs))])
    lbs, ubs = lbs / factor, ubs / factor 

    if priorData == '':
        unc_rvLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(nRxns, 1)
        priorLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(nRxns, 1)
        conLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(nRxns, 1)
        unc_rvVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(nRxns, 1)
        priorVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(nRxns, 1)
        conVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(nRxns, 1)
        unc_rvLoc = unc_rvLoc / factor
        priorLoc = priorLoc / factor
        conLoc = conLoc / factor

    else:
        with open('{0}/{1}/{2}'.format(resultDir, modelFolder, storeEPRoot), 'r') as f:
            priorDict = json.load(f)

        if len(priorDict['conLoc']) != nRxns:
            warnings.warn('Prior experimental data not found in {0}, will resort '
            'to base loc and scale initializations'.format(storeEPRoot), Warning)
            
            unc_rvLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(nRxns, 1)
            priorLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(nRxns, 1)
            conLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(nRxns, 1)
            unc_rvVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(nRxns, 1)
            priorVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(nRxns, 1)
            conVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(nRxns, 1)
            unc_rvLoc = unc_rvLoc / factor
            priorLoc = priorLoc / factor
            conLoc = conLoc / factor

        else:
            if 'unc_rvLoc' in priorDict:
                unc_rvLoc = np.array(priorDict['unc_rvLoc']).astype(dtype).reshape(
                        nRxns, 1)
                unc_rvLoc = unc_rvLoc / factor
            else:
                unc_rvLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(
                        nRxns, 1)
                unc_rvLoc = unc_rvLoc / factor
            
            if 'priorLoc' in priorDict:
                priorLoc = np.array(priorDict['priorLoc']).astype(dtype).reshape(
                        nRxns, 1)
                priorLoc = priorLoc / factor
            else:
                priorLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(
                        nRxns, 1)
                priorLoc = priorLoc / factor
            
            if 'conLoc' in priorDict:
                conLoc = np.array(priorDict['conLoc']).astype(dtype).reshape(
                        nRxns, 1)
                conLoc = conLoc / factor
            else:
                conLoc = np.repeat(locInit, nRxns).astype(dtype).reshape(nRxns, 1)
                conLoc = conLoc / factor
            
            if 'unc_rvVar' in priorDict:
                unc_rvVar = np.array(priorDict['unc_rvVar']).astype(dtype).reshape(
                        nRxns, 1)
            else:
                unc_rvVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(
                        nRxns, 1)
            
            if 'priorVar' in priorDict:
                priorVar = np.array(priorDict['priorVar']).astype(dtype).reshape(
                        nRxns, 1)
            else:
                priorVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(
                        nRxns, 1)
            
            if 'conVar' in priorDict:
                conVar = np.array(priorDict['conVar']).astype(dtype).reshape(
                        nRxns, 1)
            else:
                conVar = np.repeat(scaleInit, nRxns).astype(dtype).reshape(
                        nRxns, 1)
    
    #Ensure start locs are within bounds - necessary?
    unc_rvLoc = np.clip(unc_rvLoc, lbs, ubs)
    priorLoc = np.clip(priorLoc, lbs, ubs)
    conLoc = np.clip(conLoc, lbs, ubs)

    return (b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor,
         lbs, ubs)


def prepMatricies(S, b, priorVar, Beta):
    """ Prep matricies for EP. Inherits from runEP.
        
        Returns:
            D: Array - nRxn x nRxn matrix, with non-zero diagonals (variances) 
            KK: Array - Reaction adjacency matrix scaled by Beta
            KB: Array - Reaction vector scaled by Beta
    """
    D = np.zeros((S.shape[1], S.shape[1]), dtype)
    np.fill_diagonal(D, 1. / priorVar)
    KK = Beta * np.matmul(S.transpose(), S) 
    KB = Beta * np.matmul(S.transpose(), b)

    if (D.shape != (S.shape[1], S.shape[1]) or KK.shape != (S.shape[1], 
        S.shape[1]) or KB.shape != (S.shape[1], 1)):
        raise ValueError('Preparation of D, KK, and KB matricies failed')

    return D, KK, KB


def MetabolicEP(b, S, D, KK, KB, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, 
    conVar, lbs, ubs, factor, mseThreshold = 1e-06, minVar = 1e-50, maxVar = 1e50, 
    damp = 0.7, maxIter = 2000, GPU = True, returnCovar = False, Device = 0, 
    trackFit = False, modelName = '', **params):
    """ MetabolicEP fitting. Returns updated priors and new posteriors for loc 
        and variance within the flux bounds (constrLoc, constrLoc), in addition
        to the D diagonal matrix, which can be used to generate the Covariance
        matrix. Note that the first step is performed uniquely from the loop:
        the first matrix inversion requires integer format. EP is performed until
        either the MSE falls within the objective threshold, or the maximum
        number of iterations is hit. Optional call to track flux values at each
        EP fit iteration, although note this is not currently well-tested.

        Args:
            params: Dict - Parameter settings from .yaml
            GPU: Boolean - Use GPU for matrix inversions?
            returnCovar: Boolean - Return the final covariance matrix 
                (which requires an additional matrix inverse)
            Device: Int - Which device to perform inversions?
            trackFit: Boolean - Store results at each iteration

        Returns:
            unc_rvLoc: Array - Mean of the posterior distribution (unconstrained)
            unc_rvVar: Array - Variance of the posterior distribution (unconstrained)
            priorLoc: Array - Means of the approximated priors
            priorVar: Array - Variances of the approximated priors
            conLoc: Array - Average of the truncated Gaussians of the approximation
            conVar: Array - Variances of the truncated Gaussians of the approximation
            coVar: Array - Final covariance matrix
    """
    if trackFit == True:
        trackDict = createMetricDict(S, modelName)
    else:
        trackDict = None

    if len(set(np.diag(D))) == 1: #Check that experimental data wasnt added
        D, KK = D.astype(np.int), KK.astype(np.int) #..D is identity initially if so
        coVar = D + KK
        I1 = invertMatrix(coVar, GPU, Device, forceNp = True)
    else:
        coVar = D + KK
        I1 = invertMatrix(coVar, GPU, Device, forceNp = False)
    
    trackDict = trackLocsVars(trackDict, priorLoc, priorVar, unc_rvLoc, unc_rvVar,
            conLoc, conVar, factor, D, KK, modelName, GPU, False)

    #Mean/Var calc
    unc_rvLoc, unc_rvVar, iVar = calcMeansVars(D, I1, KB, priorLoc, priorVar, 
        unc_rvLoc, unc_rvVar, lbs, ubs, minVar, maxVar)

    #Update fluxes and moment matching
    priorLoc, priorVar, conLoc, conVar, D = updateFluxes(unc_rvLoc, unc_rvVar,
        lbs, ubs, conLoc, conVar, minVar, maxVar, iVar, damp, priorLoc, priorVar)

    trackDict = trackLocsVars(trackDict, priorLoc, priorVar, unc_rvLoc, unc_rvVar,
            conLoc, conVar, factor, D, KK, modelName, GPU, False)

    iters = 1 
    for _ in range(1, maxIter):
        iters += 1
        damp = damp - 0.001 #Adaptive damp
        if damp < 0.25:
            damp = 0.25

        if iters % 25 == 0:
            print('Iteration: {0} -- MSE: {1}'.format(iters, round(mseEstimate, 7)))
        
        coVar = D + KK
        I1 = invertMatrix(coVar, GPU, Device, forceNp = False)

        unc_rvLoc, unc_rvVar, iVar = calcMeansVars(D, I1, KB, priorLoc, priorVar, 
            unc_rvLoc, unc_rvVar, lbs, ubs, minVar, maxVar)
        
        priorLoc, priorVar, conLoc, conVar, D = updateFluxes(unc_rvLoc, 
            unc_rvVar, lbs, ubs, conLoc, conVar, minVar, maxVar, iVar, damp,
            priorLoc, priorVar)
        
        trackDict = trackLocsVars(trackDict, priorLoc, priorVar, unc_rvLoc, unc_rvVar,
            conLoc, conVar, factor, D, KK, modelName, GPU, False)
        
        mseEstimate = ((np.matmul(S, conLoc) - b)**2).mean()
        if np.isnan(mseEstimate):
            raise ArithmeticError('MetabolicEP convergence failed')

        if mseEstimate < mseThreshold:
            print('Converged after {0} steps: '.format(iters))

            if returnCovar == True:
                coVar = np.round(np.linalg.inv(D + KK), 10) #Final Covariance
                trackDict = trackLocsVars(trackDict, priorLoc, priorVar, unc_rvLoc, 
                    unc_rvVar, conLoc, conVar, factor, D, KK, modelName, GPU, True)
            
            else:
                coVar = None

            (unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc,
                conVar) = rescaleVars(unc_rvLoc, unc_rvVar, priorLoc, priorVar,
                conLoc, conVar, factor, minVar, maxVar)

            if trackFit == True:
                return unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, coVar, trackDict 
            else:
                return unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, coVar

    if iters == maxIter and mseEstimate > mseThreshold:
        raise ArithmeticError('MetabolicEP convergence failed')


def createMetricDict(S, modelName):
    """ Create a dictionary to track stats in a MetabolicEP fitting.
        Note that this function is optional, and the metrics are to be used more
        for research purposes. S is the metab x reaction metabolic matrix.
    """
    initDict = dict()
    initDict[modelName] = dict()
    for rxn in range(S.shape[1]):
        rxnName = 'v' + str(rxn)
        initDict[modelName][rxnName] = {'priorLoc': [], 'priorVar': [], 
            'unc_rvLoc': [], 'unc_rvVar': [], 'conLoc': [], 'conVar': [],
            'coVar': []}

    return initDict


def trackLocsVars(trackDict, priorLoc, priorVar, unc_rvLoc, unc_rvVar, conLoc, 
    conVar, factor, D, KK, modelName, GPU, final = False):
    """ Track updated loc and var information at each EP iteration """
    
    if trackDict == None:
        return None
    
    minVar = 1e-50 
    maxVar = 1e50
    unc_rvLoc = unc_rvLoc * factor
    unc_rvVar = np.clip(unc_rvVar * factor**2, minVar, maxVar)
    priorLoc = priorLoc * factor
    priorVar = np.clip(priorVar * factor**2, minVar, maxVar)
    conLoc = conLoc * factor
    conVar = np.clip(conVar * factor**2 , minVar, maxVar)

    if final == True:
        coVar = np.linalg.inv(D + KK)
        np.fill_diagonal(coVar, 0.)
    
    for rxn in range(len(priorLoc)):
        rxnName = 'v' + str(rxn)
        if GPU == False:
            trackDict[modelName][rxnName]['priorLoc'].append(priorLoc[rxn])
            trackDict[modelName][rxnName]['priorVar'].append(priorVar[rxn])
            trackDict[modelName][rxnName]['unc_rvLoc'].append(unc_rvLoc[rxn])
            trackDict[modelName][rxnName]['unc_rvVar'].append(unc_rvVar[rxn])
            trackDict[modelName][rxnName]['conLoc'].append(conLoc[rxn])
            trackDict[modelName][rxnName]['conVar'].append(conVar[rxn])
            if final == True:
                trackDict[modelName][rxnName]['coVar'].append(coVar[rxn, :].tolist())

        else:
            trackDict[modelName][rxnName]['priorLoc'].append(priorLoc[rxn].tolist()[0])
            trackDict[modelName][rxnName]['priorVar'].append(priorVar[rxn].tolist()[0])
            trackDict[modelName][rxnName]['unc_rvLoc'].append(unc_rvLoc[rxn].tolist()[0])
            trackDict[modelName][rxnName]['unc_rvVar'].append(unc_rvVar[rxn].tolist()[0])
            trackDict[modelName][rxnName]['conLoc'].append(conLoc[rxn].tolist()[0])
            trackDict[modelName][rxnName]['conVar'].append(conVar[rxn].tolist()[0])
            if final == True:
                trackDict[modelName][rxnName]['coVar'].append(coVar[rxn, :].tolist())

    return trackDict

def invertMatrix(coVar, GPU, Device, forceNp = False):
    """ Pull the repeated checks and if/elses out of metabolicEP and perform 
        matrix inversion in separate function.
        
        Args:
            coVar: Array - Covariance
            GPU: Boolean - Use GPU
            Device: Int - Which device to perform ops
            forceNp: Boolean - Force inversion with numpy
       
        Return:
            I1: Precision matrix
    """
    if forceNp == True:
        try:
            I1 = np.linalg.inv(coVar) 
        except np.linalg.linalg.LinAlgError:
            warnings.warn('Matrix inversion error, will resort to the '
                'Moore-Penrose pseudoinverse, however, accuracy may decrease')
            I1 = np.linalg.pinv(coVar)
        except:
            raise ArithmeticError('Matrix inversion failed')
    else:
        if GPU == False:
            try:
                I1 = np.linalg.inv(coVar) 
            except np.linalg.linalg.LinAlgError:
                warnings.warn('Matrix inversion error, will resort to the '
                    'Moore-Penrose pseudoinverse, however, accuracy may decrease')
                I1 = np.linalg.pinv(coVar)
            except:
                raise ArithmeticError('Matrix inversion failed')
        if GPU == True:
            with cp.cuda.Device(Device):
                coVar = cp.array(coVar)
                I1 = cp.linalg.inv(coVar)
                I1 = cp.asnumpy(I1) #Move back to host

    return I1


def calcMeansVars(D, I1, KB, priorLoc, priorVar, unc_rvLoc, unc_rvVar, lbs, ubs,
    minVar, maxVar):
    """ Calculate loc and variance of truncated Gaussians. Inherits from MetabolicEP """

    newLoc = np.matmul(I1, (KB + np.matmul(D, priorLoc)))
    iDiag = np.min((np.diag(I1).reshape(priorVar.shape), priorVar), axis = 0) #7x1

    priorVar[np.where((priorVar == 0.) | (np.isnan(priorVar)))] = minVar
    
    #np.maximum/minimum broadcasts properly (not np.min/max)
    iVar = np.minimum(maxVar, np.maximum(minVar, (1. / iDiag - 1. / priorVar))) 
    iVar[np.where((iVar == 0.) | (np.isnan(iVar)))] = minVar
    unc_rvVar = 1. / iVar
    
    unc_rvLoc = ((newLoc - (priorLoc * iDiag) / priorVar)) / (1 - (iDiag / priorVar))

    for rxn in range(len(unc_rvLoc)):
        if iDiag[rxn] == priorVar[rxn]: #Keep loc within bounds
            unc_rvLoc[rxn] = np.mean([lbs[rxn], ubs[rxn]])

    return unc_rvLoc, unc_rvVar, iVar


def updateFluxes(unc_rvLoc, unc_rvVar, lbs, ubs, constrLoc, constrVar, minVar, 
    maxVar, iVar, damp, priorLoc, priorVar):
    """ Calculate means/variances of titled distributions and match moments """
    
    varRoot = unc_rvVar**0.5
    momentLow = (lbs - unc_rvLoc) / varRoot 
    momentHigh = (ubs - unc_rvLoc) / varRoot #lb < ub check made upstream
    z, eps = computeMoments(momentLow, momentHigh)
    constrLoc = unc_rvLoc + (z * varRoot)
    constrVar = np.maximum(0, unc_rvVar * (1 + eps))

    #Moment updates on the priors - zeros/inf will be caught with minVar
    newVar = 1. / (1. / constrVar - 1. / unc_rvVar)
    newVar = np.minimum(maxVar, np.maximum(minVar, newVar))
    newLoc = constrLoc + (constrLoc - unc_rvLoc) * newVar * iVar 
    priorLoc = damp * priorLoc + (1 - damp) * newLoc #floor should be minVar already
    priorVar = damp * priorVar + (1 - damp) * newVar #floor should be minVar already

    priorVar[np.where((priorVar < minVar) | (np.isnan(priorVar)))] = minVar

    D = np.zeros((len(priorVar), len(priorVar)), dtype) #new covar seed
    np.fill_diagonal(D, 1. / priorVar)

    return priorLoc, priorVar, constrLoc, constrVar, D


#Prep lambda functions - faster than looping through reactions
absMinLamb = lambda x, y: min(abs(x), abs(y)) 
absMinLambV = np.vectorize(absMinLamb)
lambMean = lambda x, y: (x + y) * 0.5
lambPhi = lambda x: (1. + sciErf(x * c1)) * 0.5
lambGaussPhi = lambda x: c2 * np.exp((-x * x) * 0.5)

def computeMoments(momentLow, momentHigh):
    """ Compute moments based on tilted distributions. In order to increase speed,
        three main conditions are tested on the arrays and used to chunk the arrays
        into three pieces, so that vectorized computations can be performed on each
        piece, then rejoined at the end. Inherits from updateFluxes """

    diffs = (momentHigh - momentLow)
    cond1Index = list(np.where(diffs < 1e-10)[0]) #Condition 1
    cond1Low, cond1High = momentLow[cond1Index], momentHigh[cond1Index]
    mom1Cond1, mom2Cond1 = momentCondition1(cond1Low, cond1High)

    minLamb = absMinLambV(momentLow, momentHigh)
    prods = (momentHigh * momentLow)
    cond2Index = list(np.where((minLamb <= 6.) | (prods <= 0.))[0])
    cond2Index = list(set(cond2Index) - set(cond1Index)) #Condition 2

    if len(cond2Index) > 0:
        cond2Low, cond2High = momentLow[cond2Index], momentHigh[cond2Index]
        mom1Cond2, mom2Cond2 = momentCondition2(cond2Low, cond2High)
    else:
        mom1Cond2 = np.expand_dims(np.array([]), 1)
        mom2Cond2 = np.expand_dims(np.array([]), 1)

    addLists = set(cond1Index + cond2Index)
    allIndex = set(range(len(momentLow)))
    cond3Index = list(allIndex - addLists) #Condition 3

    if len(cond3Index) > 0:
        cond3Low, cond3High = momentLow[cond3Index], momentHigh[cond3Index]
        mom1Cond3, mom2Cond3 = momentCondition3Vec(cond3Low, cond3High)
    else:
        mom1Cond3 = np.expand_dims(np.array([]), 1)
        mom2Cond3 = np.expand_dims(np.array([]), 1)

    checkLen = len(momentLow)
    z, z1 = rejoinMoments(cond1Index, cond2Index, cond3Index, mom1Cond1, mom2Cond1, 
        mom1Cond2, mom2Cond2, mom1Cond3, mom2Cond3, checkLen) #Rejoin
    
    return z, z1


def momentCondition1(momentLowChunk, momentHighChunk):
    """ First condition, presumably on the chunked momentLow/High which meets
        criteria (delta < 1e-10). """
    mom1Cond1 = lambMean(momentLowChunk, momentHighChunk)
    mom2Cond1 = np.expand_dims(np.repeat(-1., 
        len(momentLowChunk)), 1).astype(dtype)
    return mom1Cond1, mom2Cond1


def momentCondition2(momentLowChunk, momentHighChunk):
    """ Second condition, using the pdf/cdf gaussian functions """

    erfLow = lambPhi(momentLowChunk) 
    gaussLocLow = lambGaussPhi(momentLowChunk)
    erfHigh = lambPhi(momentHighChunk) 
    gaussLocHigh = lambGaussPhi(momentHighChunk)

    mom1Cond2 = (gaussLocLow - gaussLocHigh) / (erfHigh - erfLow)
    mom2Cond2Pre = ((momentLowChunk * gaussLocLow - momentHighChunk * gaussLocHigh) /
        (erfHigh - erfLow))
    mom2Cond2 = mom2Cond2Pre - mom1Cond2 * mom1Cond2
    return mom1Cond2, mom2Cond2


def momentCondition3(momentLowChunk, momentHighChunk):
    """ Third condition, perform asymptotic expansion of moments from Brausintein
        et al """
    
    delta = 0.5 * (momentHighChunk * momentHighChunk - momentLowChunk * momentLowChunk) 
    tempVal1 = (3 - momentLowChunk * momentLowChunk + momentLowChunk ** 4)

    if delta > 40.:
            mom1Cond3 = momentLowChunk**5 / tempVal1
            mom2Cond3Pre = momentLowChunk**6 / tempVal1
    else:
            tempProd = (momentLowChunk * momentHighChunk)**5
            tempVal2 = 3 - momentHighChunk * momentHighChunk + momentHighChunk ** 4
            tempVal3 = -np.exp(delta) * tempVal1 * momentHighChunk**5
            tempVal4 = tempVal3 + momentLowChunk**5 * tempVal2
            
            mom1Cond3 = tempProd * (1. - np.exp(delta)) / tempVal4

            mom2Cond3Pre = (tempProd * (momentHighChunk - momentLowChunk * 
                np.exp(delta)) / tempVal4)

    mom2Cond3 = mom2Cond3Pre - mom1Cond3 * mom1Cond3
    return mom1Cond3, mom2Cond3


#Lot of ops to individually vectorize?
momentCondition3Vec = np.vectorize(momentCondition3)

def rejoinMoments(cond1Index, cond2Index, cond3Index, mom1Cond1, mom2Cond1, 
    mom1Cond2, mom2Cond2, mom1Cond3, mom2Cond3, checkLen):
    """ Rejoin the chunked moments and re-order based on the indicies.
        Tough to unittest, however a comparison of EP reuslts to the julia 
        "positive control" checks out (R^2 < 0.995) """

    allMom1 = np.concatenate([mom1Cond1, mom1Cond2, mom1Cond3]).astype(dtype)
    allMom2 = np.concatenate([mom2Cond1, mom2Cond2, mom2Cond3]).astype(dtype)
    if len(allMom1) != checkLen or len(allMom2) != checkLen:
        raise Exception('Rejoining of computed moments failed')
    
    origIndex = cond1Index + cond2Index + cond3Index 
    if len(origIndex) != checkLen:
        raise Exception('Rejoining of moment indicies failed')

    z = allMom1[np.argsort(origIndex)]
    z1 = allMom2[np.argsort(origIndex)]

    return z, z1


def rescaleVars(unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, 
    factor, minVar, maxVar):
    """ Return to original scale. Inherits from runEP """

    unc_rvLoc = unc_rvLoc * factor
    unc_rvVar = np.clip(unc_rvVar * factor**2, minVar, maxVar)
    priorLoc = priorLoc * factor
    priorVar = np.clip(priorVar * factor**2, minVar, maxVar)
    conLoc = conLoc * factor
    conVar = np.clip(conVar * factor**2 , minVar, maxVar)

    return unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar


def storeMetabolicEP(modelName, tag, unc_rvLoc, unc_rvVar, priorLoc, priorVar, 
    conLoc, conVar, expParams = None, baseDict = 'modelDict.json', 
    modelPath = 'ref', resultDir = 'resultStore', subResultFolder = '',
    storeEPRoot = 'storeEP.json', overwrite = False, **params):
    """ Store MetabolicEP results. Any experimental conditions for the data can
        be stored in expParams and written with the EP results. The name of the
        study, or perhaps the sample ID from GEO, can be specified in the file
        name with the "tag" argument. Note that default conditions are hard-coded
        in this function, as it is unclear as to whether or not this will need
        to be changed in the future (or even used much).

        Args:
            modelName: Str - Name of metabolic model
            tag: Str - Experiment identifier, as part of the file name. Could come
                from the sample ID in GEO, for instance.
            expParams: Dict - Dictionary of experimental conditions to store with
                the MetabolicEP results. If none, a default set of values will be
                saved.
            params: Dict - Parameter settings from .yaml
            baseDict: Str - Base model reference dictionary
            modelPath: Str - Location of Metabolic models
            resultDir: Str - Directory of stored EP results
            storeEPRoot: Str - json of stored EP results
            overwrite: Boolean - Overwrite entries of existing storeEPRoot?
        
        Returns:
            EP results stored in .json under the resultDir
    """
    if tag == '':
        raise ValueError('Tag must be a real string in order to write out results')

    if re.search(r'\ ', tag):
        warnings.warn('Tag ID contains spaces, will remove for file write safety',
             Warning)
        tag = re.sub(r'\ ', '', tag)

    modelFolder = re.sub(r'\ |\.', '', modelName)
    if subResultFolder != '':
            modelFolder = modelFolder + '/' + subResultFolder
    resultFile = tag + '_' + storeEPRoot 

    if resultFile not in os.listdir('{0}/{1}'.format(resultDir, modelFolder)):
        checkFin = True
    else:
        checkModelPresence(modelName, modelFolder, baseDict, modelPath, resultDir, 
            resultFile, overwrite)
        checkFin = False

    baseCondition = dict({'GEO': '', 'ID': '', 'Description': '', 
        'Characteristics': '', 'Organism': '', 'Tissue': '', 'SampleType': '', 
        'KO': None, 'Expression': False, 'Thermodynamics': False, 
        'Variants': False, 'Protein': False, 'Metabolite': False})

    if isinstance(expParams, dict):
        if len(expParams) == 0:
            raise Exception('Experimental parameter dictionary is empty')
        diff = [x for x in baseCondition if x not in expParams]
        if len(diff) == len(baseCondition):
            raise Exception('Experimental parameter dictionary contains none of '
                'the expected keys')
    else:
        print('No experimental conditions detected, will resort to "base" '
            'parameter values')
        expParams = baseCondition 
    
    modelDict = expParams
    length = len(unc_rvLoc)
    modelDict['unc_rvLoc'] = unc_rvLoc.reshape(length, ).tolist()
    modelDict['unc_rvVar'] = unc_rvVar.reshape(length, ).tolist()
    modelDict['priorLoc'] = priorLoc.reshape(length, ).tolist()
    modelDict['priorVar'] = priorVar.reshape(length, ).tolist()
    modelDict['conLoc'] = conLoc.reshape(length, ).tolist()
    modelDict['conVar'] = conVar.reshape(length, ).tolist()

    with open('{0}/{1}/{2}'.format(resultDir, modelFolder, resultFile), 'w') as fOut:
        json.dump(modelDict, fOut, indent = 4, sort_keys = False)
        fOut.write('\n')

    if checkFin == True:
        checkModelPresence(modelName, modelFolder, baseDict, modelPath, resultDir, 
            resultFile, overwrite, suppress = True)


def checkModelPresence(modelName, modelFolder, baseDict, modelPath, resultDir, 
    resultFile, overwrite, suppress = False):
    """ Check the presence of a specified metabolic model in the model and 
        EP result storage repositories. This function only inherits from other
        functions (separated here to save redundant code). """

    with open('{0}/{1}'.format(modelPath, baseDict), 'r') as fIn:
        refDict = json.load(fIn)

    if modelName not in refDict:
        raise KeyError('{0} not found in {1}: Add it by using '
            'updateModelDict() in setup_gemUtl.py'.format(modelName, baseDict))
   
    modelDirectory = '{0}/{1}'.format(resultDir, modelFolder)
    if resultFile not in os.listdir(modelDirectory):
        raise FileExistsError('{0} not found under {1} directory: Add it by using '
            'buildEPStorageFromRef()'.format(resultFile, modelDirectory))
    
    if resultFile in os.listdir(modelDirectory) and overwrite == False:
        raise PermissionError('Overwrite on existing {0} disallowed'.format(resultFile))
    elif resultFile in os.listdir(modelDirectory) and overwrite == True and suppress == False:
        warnings.warn('You elected to overwrite results in {0}'.format(resultFile))


def receiveGEOWrangleDict(EPDict, useInternalPriors = True,
    baseDict = 'modelDict.json', modelPath = 'ref', resultDir = 'resultStore', 
    storeEPRoot = 'storeEP.json', minVar = 1e-50, maxVar = 1e50, damp = 0.7, 
    mseThreshold = 1e-06, Beta = 1e07, maxIter = 1000, locInit = 0.0, 
    scaleInit = 0.5, GPU = True, Device = 0, returnCovar = False, 
    overwrite = False, **params):
    """ Take the epInfoDict from prepEPDict() in GEOWrangle functions. The 
        essential elements here are extracting flux bounds, setting up the
        storage directory and storeEP file, and returning the variables which can
        be plugged directly into MetabolicEP functions. This function is a critcal
        handle between GEOWrangle and MetabolicEP. 

        Args:
            EPDict: Dict - Bounds and conditions dictionary from GEOWrangle
            params: Dict - Parameter settings from .yaml file
            useInternalPriors: Boolean - Use priors from within the storeEPDir
                field in the EPDict (e.g. one sample's solution as priors for the
                other samples within a GSE)

        Returns:
            S, lbs, ubs params: MetabolicEP inputs
            tag, expParams: MetabolicEP storage and condition information
    """
    storeRef = EPDict['MetaData']['Store'] #e.g. 'GSE' for a GEO study
    if storeRef not in EPDict['MetaData']:
        raise Exception('Sub-directory for EP storage not found in EPDict')
    
    subResultFolder = EPDict['MetaData'][storeRef]
    modelName = EPDict['MetaData']['modelName']
    modelFolder = re.sub(r'\ |\.', '', modelName)
    
    with open('{0}/{1}'.format(modelPath, baseDict), 'r') as fIn:
        refDict = json.load(fIn)

    if modelName not in refDict:
        raise KeyError('{0} not found in {1}: Add it by using '
            'updateModelDict() in setup_gemUtl.py'.format(modelName, baseDict))

    sumPath = '{0}/{1}'.format(resultDir, modelFolder)
    fullPath = '{0}/{1}'.format(sumPath, subResultFolder)
    if subResultFolder not in os.listdir(sumPath):
        os.makedirs(fullPath)
    
    priorsFiles = [x for x in os.listdir(fullPath) if storeEPRoot in x]
    if len(os.listdir(fullPath)) > 0 and useInternalPriors == True and len(priorsFiles) > 0:
        priorEP = random.choice(priorsFiles)
        priorEP = re.sub('\_{0}'.format(storeEPRoot), '', priorEP)
    elif 'Base_{0}'.format(storeEPRoot) in os.listdir(sumPath): #use Base
        print('useInternalPriors is set to True, but no other results under {0} '
            'exist, resorting to base'.format(fullPath))
        priorEP = 'Base'
    else:
        priorEP = ''

    #2 Get bounds
    S, lbs, ubs = getGEOWrangleData(EPDict, modelName, refDict)

    #3 Build tag, params, and expParams(for storeMetabolicEP)
    tag, params, expParams = getGEOWrangleParams(EPDict, params, storeEPRoot, 
        subResultFolder, priorEP, modelPath, baseDict, resultDir, minVar, 
        maxVar, damp, mseThreshold, Beta, maxIter, locInit, scaleInit, GPU, 
        Device, returnCovar, overwrite)

    return S, lbs, ubs, tag, params, expParams


def getGEOWrangleData(EPDict, modelName, refDict):
    """ Extract S, lbs, and ubs from the EP dict prepared in GEOWrangle. Inherits
        from receiveGEOWrangleDict """
    
    S = np.load(refDict[modelName]['S']).astype(dtype)
    varShape = [S.shape[1], 1]
    lbs = []
    ubs = []
    for rxn in EPDict['Data']:
        lbs.append(EPDict['Data'][rxn]['lb'])
        ubs.append(EPDict['Data'][rxn]['ub'])  
    lbs = np.array(lbs).astype(dtype).reshape(varShape)
    ubs = np.array(ubs).astype(dtype).reshape(varShape)

    return S, lbs, ubs


def getGEOWrangleParams(EPDict, params, storeEPRoot, subResultFolder, priorEP, 
    modelPath, baseDict, resultDir, minVar, maxVar, damp, mseThreshold, Beta,
    maxIter, locInit, scaleInit, GPU, Device, returnCovar, overwrite):
    """ Extract parameters for metabolicEP from GEOWrangle. Inherits from 
        receiveGEOWrangleDict. Note that the device placement is directly read
        from the EPDict: Device is one of few MetabolicEP-specific arguments
        that can be called from GEOWrangle (the others, such as locInit, Beta, 
        etc, are best left to within-MetabolicEP params). """

    tag = EPDict['MetaData']['tag']
    expParams = EPDict['MetaData']

    if params is not None:
        params['subResultFolder'] = subResultFolder
        params['priorData'] = priorEP
        params['Device'] = EPDict['MetaData']['Device']
        return tag, params, expParams

    else: #build the params for EP
        params = dict()
        params['modelPath'] = modelPath
        params['baseDict'] = baseDict
        params['resultDir'] = resultDir
        params['storeEPRoot'] = storeEPRoot
        params['subResultFolder'] = subResultFolder
        params['minVar'] = minVar
        params['maxVar'] = maxVar
        params['damp'] = damp
        params['mseThreshold'] = mseThreshold
        params['Beta'] = Beta
        params['maxIter'] = maxIter
        params['locInit'] = locInit
        params['scaleInit'] = scaleInit
        params['priorData'] = priorEP
        params['GPU'] = GPU
        params['Device'] = Device
        params['returnCovar'] = returnCovar
        params['overwrite'] = overwrite 
        return tag, params, expParams
