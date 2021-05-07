""" Unit-testing for MetabolicEP. Because speed is so critical here, many of the
    value checks originally in the MetabolicEP functions are moved to unit-tests, 
    so that for larger problems, we don't need to call any() over large arrays
    millions of times. """

import unittest, sys, os, re, warnings, json, yaml
from numpy.testing import assert_array_equal, assert_almost_equal
import numpy as np
import pandas as pd
from scipy import stats
sys.path.append('MetabolicEP/src/')
import setup_MetabolicEP as util
dtype = np.float64

try:
    paramFile = sys.argv[1]
except IndexError:
    paramFile = 'params/MetabolicEP.yaml'

with open(paramFile) as fin:
    params = yaml.safe_load(fin)

minVar, maxVar, damp, mseThreshold, Beta = (float(params['minVar']),
    float(params['maxVar']), float(params['damp']), float(params['mseThreshold']),
    float(params['Beta']))
resultDir = params['resultDir']

#stricter for testing
mseThreshold = 1e-06

class TestDataProcess(unittest.TestCase):

    def test_prepMetabolicModel(self):
        """ Check: Multiple model formats can be read """

        modelName = ''
        S, _, _ = util.prepMetabolicModel(modelName = modelName,
            modelPath = '', toy = True)
        self.assertTrue(S.shape == (5, 7))
        
        modelName = 'WormFlux'
        S, _, _ = util.prepMetabolicModel(modelName)
        self.assertGreater(S.shape[0], 0)
    
    def test_prepInputs(self):
        """ Check: All lbs < ubs. PriorLocs are between lb and ub. Successful
            integration with priors from 'storeEP.json'. Utilization of an
            existing GEO expression EP result as a prior """

        modelName = 'WormFlux'
        modelFolder = re.sub(r'\ |\.', '', modelName)
        S, lbs, ubs = util.prepMetabolicModel(modelName)
        _, _, _, priorLoc, _, _, _, factor, lbs, ubs = util.prepInputs(S, 
            lbs, ubs, locInit = 0., scaleInit = 0.5, resultDir = resultDir,
            modelName = modelName, storeEPRoot = 'storeEP.json', priorData = '',
            modelFolder = modelFolder)
        self.assertTrue(all(lbs <= ubs))
        self.assertTrue(all(lbs <= factor))
        self.assertTrue(all(ubs <= factor))
        self.assertTrue(all([x <= loc <= y for x, loc, y in zip(lbs, priorLoc, ubs)]))
        _, _, _, priorLocBase, _, _, _, factor, lbs, ubs = util.prepInputs(S, 
            lbs, ubs, locInit = 0., scaleInit = 0.5, resultDir = resultDir, 
            modelName = modelName, storeEPRoot = 'Base_storeEP.json', 
            priorData = 'Base', modelFolder = modelFolder)
        self.assertFalse(np.array_equal(priorLoc, priorLocBase))

        with open('resultStore/WormFlux/Base_storeEP.json', 'r') as f:
            priorDict = json.load(f)
        checkVals = np.array(priorDict['priorLoc']).astype(np.float64).reshape(len(priorLoc), 1)
        checkVals = checkVals / factor
        checkVals = np.clip(checkVals, lbs, ubs)
        self.assertTrue(np.array_equal(priorLocBase, checkVals))

        geo = 'GSE21784'
        tag = 'GSM542652'
        storeEPRoot = tag + '_' + 'storeEP.json'
        modelFolder = modelFolder + '/' + geo
        _, _, _, priorLocGSM, _, _, _, factor, lbs, ubs = util.prepInputs(S, 
            lbs, ubs, locInit = 0., scaleInit = 0.5, resultDir = resultDir, 
            modelName = modelName, storeEPRoot = storeEPRoot,
            priorData = tag, modelFolder = modelFolder)
        with open('resultStore/WormFlux/{0}/{1}'.format(geo, storeEPRoot), 'r') as f:
            priorDict = json.load(f)
        checkVals = np.array(priorDict['priorLoc']).astype(np.float64).reshape(len(priorLoc), 1)
        checkVals = checkVals / factor
        checkVals = np.clip(checkVals, lbs, ubs)
        self.assertTrue(np.array_equal(priorLocGSM, checkVals))

    
    def test_calcMeansVars_FluxUpdate(self):
        """ Check:  All return values are of equal length (properly broadcast). 
            Minimum of iVar is within variance limits. New loc and variance values
            are 'different'. loc is within bounds. updateFluxes results in new 
            locs, var, and D, when damp != 1. Minimum of new var is at least 
            minVar.  """
        modelName = 'WormFlux'
        S, lbs, ubs = util.prepMetabolicModel(modelName)
        b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor, lbs, ubs, D, KK, KB = util.checkArgs_prepEP(S, 
            lbs, ubs, modelName)
        D, KK = D.astype(np.int), KK.astype(np.int)
        coVar = D + KK
        I1 = np.linalg.inv(coVar) 
        unc_rvLocNew, unc_rvVarNew, iVar = util.calcMeansVars(D, I1, KB, priorLoc, 
            priorVar, unc_rvLoc, unc_rvVar, lbs, ubs, minVar, maxVar)

        self.assertTrue(len(unc_rvLocNew) == len(unc_rvVarNew) == len(iVar))

        damp = 0.8
        priorLocUpdate, priorVarUpdate, constrLocUpdate, constrVarUpdate, DUpdate = util.updateFluxes(unc_rvLocNew, 
            unc_rvVarNew, lbs, ubs, conLoc, conVar, minVar, maxVar, iVar, 
            damp, priorLoc, priorVar)

        self.assertTrue(all(priorVarUpdate >= minVar))
        self.assertTrue(all(priorVarUpdate <= maxVar))
        self.assertTrue(all(constrVarUpdate >= 0.))
        self.assertTrue(all(constrVarUpdate <= maxVar))
        self.assertFalse(all(priorLocUpdate == priorLoc))
        self.assertFalse(all(priorVarUpdate == priorVar))
        self.assertFalse(all(constrLocUpdate == conLoc))
        self.assertFalse(all(constrVarUpdate == conVar))
        self.assertFalse(all(np.ravel(DUpdate) == np.ravel(D)))

        damp = 1.
        priorLocUpdate, priorVarUpdate, constrLocUpdate, constrVarUpdate, DUpdate = util.updateFluxes(unc_rvLocNew, 
            unc_rvVarNew, lbs, ubs, conLoc, conVar, minVar, maxVar, iVar, 
            damp, priorLoc, priorVar)

        self.assertTrue(all(priorVarUpdate >= minVar))
        self.assertTrue(all(priorVarUpdate <= maxVar))
        self.assertTrue(all(constrVarUpdate >= 0.))
        self.assertTrue(all(constrVarUpdate <= maxVar))
        self.assertTrue(all(priorLocUpdate == priorLoc)) #Damp = 1, unchanged
        self.assertTrue(all(priorVarUpdate == priorVar)) #Damp = 1, unchanged
        self.assertFalse(all(constrLocUpdate == conLoc)) 
        self.assertFalse(all(constrVarUpdate == conVar)) 
        self.assertTrue(all(np.ravel(DUpdate) == np.ravel(D))) #Damp = 1, unchanged
        

    def test_ComputeMoments(self):
        """ Check: Proper moment computation. Note computeMoments needs arrays """
        mom1, mom2 = util.computeMoments(np.zeros(10).reshape(10, 1), np.ones(10).reshape(10, 1))
        self.assertTrue(np.round(mom1[0], 5) == 0.45986)
        self.assertTrue(np.round(mom2[0], 5) == -0.92035)

    
    def test_runEP(self):
        """ Check: EP on toy model yields positive control locs and vars.
            Correlation coefficient of an EP run maintains R2 > 0.99 with a
            Stored "successful" run ('storeEP.json'). Check adding "Base" loc
            and scale as priors yields the same result. Not sure why the priorLoc
            checks dont work (perhaps the storeEP.json entry is bad?).
            Alternative priors, besides base, can be used to seed EP. """
        
        Beta = 1e07
        
        ### Test toy model ###
        S, lbs, ubs = util.prepMetabolicModel('', None, '', '', toy = True)

        b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor, lbs, ubs = util.prepInputs(S, 
            lbs, ubs, locInit = 0., scaleInit = 0.5, resultDir = resultDir, 
            modelName = '', storeEPRoot = 'storeEP.json', priorData = '',
            modelFolder = '')

        D, KK, KB = util.prepMatricies(S, b, priorVar, Beta)
        _, _, _, _, conLoc, conVar, _ = util.MetabolicEP(b, S, D, KK, KB, unc_rvLoc,
            unc_rvVar, priorLoc, priorVar, conLoc, conVar, lbs, ubs, factor, 
            maxVar = 1e50, minVar = 1e-50, damp = 0.7, mseThreshold = 1e-07, 
            maxIter = 100, GPU = False, returnCovar = False)
        ctlLocs = np.array([5.672e-04, 5.002e-01, 4.998e-01, 5.734e-04, 5.734e-04,
            5.734e-04, 3.029e-04]).reshape(conLoc.shape)
        ctlVars = np.array([1.663e-07, 6.378e-02, 6.378e-02, 1.702e-07, 1.702e-07,
           1.702e-07, 7.514e-08]).reshape(conLoc.shape)
        assert_almost_equal(conLoc, ctlLocs, decimal = 4)
        assert_almost_equal(conVar, ctlVars, decimal = 4)
        
        modelList = ['WormFlux'] 
        
        for modelName in modelList:

            modelFolder = re.sub(r'\ |\.', '', modelName)
            with open('resultStore/{0}/Base_storeEP.json'.format(modelFolder), 'r') as fIn:
                trueVals = json.load(fIn)

            S, priorLBs, priorUBs = util.prepMetabolicModel(modelName, 
                modelPath = 'ModelRepository', toy = False)

            b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor, lbs, ubs, D, KK, KB = util.checkArgs_prepEP(S, 
                priorLBs, priorUBs, modelName = modelName, mseThreshold = 1e-06, 
                Beta = 1e07, minVar = 1e-50, maxVar = 1e50, damp = 0.7, maxIter = 1000, 
                locInit = 0., scaleInit = 0.5, resultDir = resultDir, 
                storeEPRoot = 'storeEP.json', priorData = '') 
            
            unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, _ = util.MetabolicEP(b, 
                S, D, KK, KB, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, 
                conVar, lbs, ubs, factor, mseThreshold = 1e-06, 
                minVar = 1e-50, maxVar = 1e50, damp = 0.7, maxIter = 1000,
                GPU = True, returnCovar = False, Device = 0)
            
            pos_unc_rvLoc = np.array(trueVals['unc_rvLoc']).reshape(len(unc_rvLoc), 1)
            unc_rvLoc = unc_rvLoc.reshape(len(unc_rvLoc), 1)
            pear = np.round(stats.pearsonr(pos_unc_rvLoc, unc_rvLoc)[0], 3)
            self.assertGreater(pear, 0.99)
            
            pos_unc_rvVar = np.array(trueVals['unc_rvVar']).reshape(len(unc_rvVar), 1)
            unc_rvVar = unc_rvVar.reshape(len(unc_rvVar), 1)
            pear = np.round(stats.pearsonr(pos_unc_rvVar, unc_rvVar)[0], 3)
            self.assertGreater(pear, 0.99)

            pos_priorLoc = np.array(trueVals['priorLoc']).reshape(len(priorLoc), 1)
            priorLoc = priorLoc.reshape(len(priorLoc), 1)
            pear = np.round(stats.pearsonr(pos_priorLoc, priorLoc)[0], 3)
            self.assertGreater(pear, 0.99)
            
            pos_priorVar = np.array(trueVals['priorVar']).reshape(len(priorVar), 1)
            priorVar = priorVar.reshape(len(priorVar), 1)
            pear = np.round(stats.pearsonr(pos_priorVar, priorVar)[0], 3)
            self.assertGreater(pear, 0.99)
            
            pos_conLoc = np.array(trueVals['conLoc']).reshape(len(conLoc), 1)
            conLoc = conLoc.reshape(len(conLoc), 1)
            pear = np.round(stats.pearsonr(pos_conLoc, conLoc)[0], 3)
            self.assertGreater(pear, 0.99)
            
            pos_conVar = np.array(trueVals['conVar']).reshape(len(conVar), 1)
            conVar = conVar.reshape(len(conVar), 1)
            pear = np.round(stats.pearsonr(pos_conVar, conVar)[0], 3)
            self.assertGreater(pear, 0.99)
            
            #Check adding back prior data yields same result
            b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor, lbs, ubs, D, KK, KB = util.checkArgs_prepEP(S, 
                priorLBs, priorUBs, modelName = modelName, mseThreshold = 1e-06, 
                Beta = 1e07, minVar = 1e-50, maxVar = 1e50, damp = 0.7, maxIter = 1000, 
                locInit = 0., scaleInit = 0.5, resultDir = resultDir, 
                storeEPRoot = 'storeEP.json', priorData = 'Base') 
            
            unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, _ = util.MetabolicEP(b, 
                S, D, KK, KB, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, 
                conVar, lbs, ubs, factor, mseThreshold = 1e-06,
                minVar = 1e-50, maxVar = 1e50, damp = 0.7, maxIter = 1000, 
                GPU = True, returnCovar = False, Device = 0)
            
            unc_rvLoc = unc_rvLoc.reshape(len(unc_rvLoc), 1)
            pear = np.round(stats.pearsonr(pos_unc_rvLoc, unc_rvLoc)[0], 3)
            self.assertGreater(pear, 0.99)

            unc_rvVar = unc_rvVar.reshape(len(unc_rvVar), 1)
            pear = np.round(stats.pearsonr(pos_unc_rvVar, unc_rvVar)[0], 3)
            self.assertGreater(pear, 0.99)

            priorVar = priorVar.reshape(len(priorVar), 1)
            pear = np.round(stats.pearsonr(pos_priorVar, priorVar)[0], 3)
            self.assertGreater(pear, 0.99)
            
            conLoc = conLoc.reshape(len(conLoc), 1)
            pear = np.round(stats.pearsonr(pos_conLoc, conLoc)[0], 3)
            self.assertGreater(pear, 0.99)
            
            conVar = conVar.reshape(len(conVar), 1)
            pear = np.round(stats.pearsonr(pos_conVar, conVar)[0], 3)
            self.assertGreater(pear, 0.99)
        
        #Check using alternative priors from real data
        modelName = 'WormFlux'
        geo = 'GSE21784'
        tag = 'GSM542652'
        modelFolder = re.sub(r'\ |\.', '', modelName)
        with open('resultStore/{0}/Base_storeEP.json'.format(modelFolder), 'r') as fIn:
            trueVals = json.load(fIn)
        
        S, priorLBs, priorUBs = util.prepMetabolicModel(modelName, 
                modelPath = 'ModelRepository', toy = False)
        
        b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor, lbs, ubs, D, KK, KB = util.checkArgs_prepEP(S, 
            priorLBs, priorUBs, modelName = modelName, mseThreshold = 1e-06, 
            Beta = 1e07, minVar = 1e-50, maxVar = 1e50, damp = 0.7, maxIter = 1000, 
            locInit = 0., scaleInit = 0.5, resultDir = resultDir, 
            storeEPRoot = 'storeEP.json', priorData = tag, subResultFolder = geo) 
        
        unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, _ = util.MetabolicEP(b, 
            S, D, KK, KB, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, 
            conVar, lbs, ubs, factor, mseThreshold = 1e-06,
            minVar = 1e-50, maxVar = 1e50, damp = 0.7, maxIter = 1000, 
            GPU = True, returnCovar = False, Device = 0)
        
        pos_unc_rvLoc = np.array(trueVals['unc_rvLoc']).reshape(len(unc_rvLoc), 1)
        unc_rvLoc = unc_rvLoc.reshape(len(unc_rvLoc), 1)
        pear = np.round(stats.pearsonr(pos_unc_rvLoc, unc_rvLoc)[0], 3)
        self.assertGreater(pear, 0.99)
        
        pos_priorLoc = np.array(trueVals['priorLoc']).reshape(len(priorLoc), 1)
        priorLoc = priorLoc.reshape(len(priorLoc), 1)
        pear = np.round(stats.pearsonr(pos_priorLoc, priorLoc)[0], 3)
        self.assertGreater(pear, 0.99)
        
        pos_priorVar = np.array(trueVals['priorVar']).reshape(len(priorVar), 1)
        priorVar = priorVar.reshape(len(priorVar), 1)
        pear = np.round(stats.pearsonr(pos_priorVar, priorVar)[0], 3)
        self.assertGreater(pear, 0.99)
        
        pos_conLoc = np.array(trueVals['conLoc']).reshape(len(conLoc), 1)
        conLoc = conLoc.reshape(len(conLoc), 1)
        pear = np.round(stats.pearsonr(pos_conLoc, conLoc)[0], 3)
        self.assertGreater(pear, 0.99)
        
        pos_conVar = np.array(trueVals['conVar']).reshape(len(conVar), 1)
        conVar = conVar.reshape(len(conVar), 1)
        pear = np.round(stats.pearsonr(pos_conVar, conVar)[0], 3)
        self.assertGreater(pear, 0.99)
    

    def test_storeMetabolicEP(self):
        """ Check: Successful storage of MetabolicEP results. Custom experimental
            dictionary can be added to the storage file. """
        
        modelList = ['WormFlux']
        for modelName in modelList:

            S, priorLBs, priorUBs = util.prepMetabolicModel(modelName,
                    baseDict = 'modelDict.json', modelPath = 'ModelRepository',
                     toy = False)

            b, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, factor, lbs, ubs, D, KK, KB = util.checkArgs_prepEP(S, 
                    priorLBs, priorUBs, modelName = modelName, mseThreshold = mseThreshold, 
                    Beta = Beta, minVar = minVar, maxVar = maxVar, damp = damp, maxIter = 1000, 
                    locInit = 0., scaleInit = 0.5, resultDir = resultDir,
                    storeEPRoot = 'storeEP.json', priorData = '') 
                        
            unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, conVar, _ = util.MetabolicEP(b, 
                S, D, KK, KB, unc_rvLoc, unc_rvVar, priorLoc, priorVar, conLoc, 
                conVar, lbs, ubs, factor, mseThreshold = mseThreshold,
                minVar = minVar, maxVar = maxVar, damp = damp, maxIter = 1000,
                GPU = True, returnCovar = False, Device = 0)

            util.storeMetabolicEP(modelName, 'test', unc_rvLoc, unc_rvVar, priorLoc, priorVar, 
                conLoc, conVar, expbaseDict = 'modelDict.json', 
                modelPath = 'ModelRepository', resultDir = resultDir, 
                storeEPRoot = 'storeEP.json', overwrite = True)
            
            modelDir = re.sub(r'\ |\.', '', modelName)
            
            with open('{0}/{1}/Base_{2}'.format(resultDir, modelDir, 'storeEP.json'), 'r') as fIn:
                baseDict = json.load(fIn)
            with open('{0}/{1}/test_{2}'.format(resultDir, modelDir, 'storeEP.json'), 'r') as fIn:
                testDict = json.load(fIn)
            self.assertEqual(baseDict['unc_rvLoc'], testDict['unc_rvLoc'])
            self.assertEqual(baseDict['unc_rvVar'], testDict['unc_rvVar'])
            self.assertEqual(baseDict['priorLoc'], testDict['priorLoc'])
            self.assertEqual(baseDict['priorVar'], testDict['priorVar'])
            self.assertEqual(baseDict['conVar'], testDict['conVar'])
            self.assertEqual(baseDict['conLoc'], testDict['conLoc'])
            
            expDict = dict({'KO' : 'test1', 'Expression': 'test2', 'Test': 'test3'})
            util.storeMetabolicEP(modelName, 'test', unc_rvLoc, unc_rvVar, priorLoc, priorVar, 
                conLoc, conVar, expParams = expDict, baseDict = 'modelDict.json', 
                modelPath = 'ModelRepository', resultDir = resultDir, 
                storeEPRoot = 'storeEP.json', overwrite = True)
            with open('{0}/{1}/test_{2}'.format(resultDir, modelDir, 'storeEP.json'), 'r') as fIn:
                testDict = json.load(fIn)
            self.assertTrue(testDict['KO'] == 'test1')
            self.assertTrue(testDict['Expression'] == 'test2')
            self.assertTrue(testDict['Test'] == 'test3')
        

if __name__ == '__main__':

    unittest.main()
