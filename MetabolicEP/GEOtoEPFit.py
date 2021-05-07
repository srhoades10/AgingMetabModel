""" Feed GEOWrangle to MetabolicEP. The intended use of this script is to 
    call it from bash script (e.g. via subprocess), thus the dictionary 
    containing both the data and conditions for MetabolicEP must be provided in
    the shell command. It is not recommended to call this function directly, or 
    solely run MetabolicEP with it. The necessary information is housed in the
    dictionary, and functions within the MetabolicEP module are designed to 
    extract everything locally. """


import json, sys, yaml, os

epDirectory = sys.argv[1]
epDictFile = sys.argv[2]
defaultParams = bool(sys.argv[3])
try:
    resultDirTag = sys.argv[4]
except IndexError:
    resultDirTag = ''
paramFile = 'params/MetabolicEP.yaml'

os.chdir('../{0}'.format(epDirectory))
sys.path.append('./src/')
import setup_MetabolicEP as util

def main(epDictFile, defaultParams, paramFile, resultDirTag):

    if defaultParams == True:
        with open(paramFile) as fin:
            params = yaml.safe_load(fin)
    params['resultDir'] = params['resultDir'] + resultDirTag

    with open(epDictFile, 'r') as fin:
        epDict = json.load(fin)

    try:
        S, priorLBs, priorUBs, tag, params, expParams = util.receiveGEOWrangleDict(epDict, 
            useInternalPriors = True, **params)

        modelName = epDict['MetaData']['modelName']
        S, _, _ = util.prepMetabolicModel(modelName = modelName,
                **params) 
        
        (b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor,
            lbs, ubs, D, KK, KB) = util.checkArgs_prepEP(S, priorLBs, priorUBs, 
            modelName = modelName, **params) 
                
        (unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, 
            conVar, coVar) = util.MetabolicEP(b, S, D, KK, KB, unc_rvLoc, 
            unc_rvVar, priorLoc, priorVar, conLoc, conVar, lbs, ubs, factor, 
            **params)

        util.storeMetabolicEP(modelName, tag, unc_rvLoc, unc_rvVar, 
            priorLoc, priorVar, conLoc, conVar, expParams = expParams, **params)

    except ArithmeticError:
        print('Convergence failed, will attempt MetabolicEP again without priors')
        
        S, priorLBs, priorUBs, tag, params, expParams = util.receiveGEOWrangleDict(epDict, 
            useInternalPriors = False, **params)

        modelName = epDict['MetaData']['modelName']
        S, _, _ = util.prepMetabolicModel(modelName = modelName,
                **params) 
        
        (b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor,
            lbs, ubs, D, KK, KB) = util.checkArgs_prepEP(S, priorLBs, priorUBs, 
            modelName = modelName, **params) 
                
        (unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, 
            conVar, coVar) = util.MetabolicEP(b, S, D, KK, KB, unc_rvLoc, 
            unc_rvVar, priorLoc, priorVar, conLoc, conVar, lbs, ubs, factor, 
            **params)

        util.storeMetabolicEP(modelName, tag, unc_rvLoc, unc_rvVar, 
            priorLoc, priorVar, conLoc, conVar, expParams = expParams, **params)


if __name__ == '__main__':

    main(epDictFile, defaultParams, paramFile, resultDirTag)

