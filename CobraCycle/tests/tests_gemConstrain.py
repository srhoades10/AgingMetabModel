""" Unit-testing for gem constraints. """

import unittest, sys, os, re, warnings, cobra, yaml, random, json

sys.path.append('CobraCycle/src/')
import setup_gemConstrain as gem
import setup_GEOWrangle as geo
import setup_DataNormalization as util
modelDir = 'CobraCycle/ref'
paramFile = 'CobraCycle/params/GEO_GeneExprProcess.yaml'
wormECMatchFile, mouseECMatchFile = 'WormFluxECMatch.json', 'MouseECMatch.json'
kcatFile = 'EColi_GoldStanKCats.json'

with open(paramFile) as fin:
    params = yaml.safe_load(fin)
with open(params['modelDictFile']) as fin:
    modelDict = json.load(fin)

class TestDataProcess(unittest.TestCase):
    
    def test_rxnIndexExtract(self):
        """ Check: multiType > singleType. 
            Type == 'multicompartment' == rxnInfo['multicompartment']
            Type == 'exchanges' == rxnInfo['exchanges']. Successful compartName
            conversion ('Extracellular' == 'extracellular' == 'e'). Valid
            subsystem name. multiCompartSub returns same as 'multicompartment' if
            all compartments are specified. 
            compartName + multiCompartSub > compartName.
        """

        for mods in ['MODEL1507180055.xml']:

            modelRead = cobra.io.read_sbml_model('{0}/{1}'.format(modelDir, mods))
        
            rxnInfo = gem.modelReactionInfo(modelRead)
            set1 = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['compartment'], 
                compartName = 'e', subSysName = None)
            set1a = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['compartment'], 
                compartName = 'E', subSysName = None)
            self.assertEqual(len(set1), len(set1a))
            set2 = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['compartment', 
                'subsystem'], compartName = 'e', subSysName = None)
            set3 = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['compartment', 
                'subsystem'], compartName = 'e', subSysName = 'Pyruvate Metabolism')
            set4 = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['compartment', 
                'subsystem'], compartName = 'e', subSysName = 'pyruvate metabolism')
            self.assertEqual(len(set2), len(set1))

            set5a = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['multicompartment'])
            self.assertEqual(len(set5a), len(rxnInfo['multicompartment']))
            set5b = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['exchanges'])
            self.assertEqual(len(set5b), len(rxnInfo['exchanges']))

            set6 = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['compartment'], 
                compartName = 'Extracellular', subSysName = None)
            set7 = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['compartment'], 
                compartName = 'extracellular', subSysName = None)
            self.assertEqual(len(set6), len(set1))
            self.assertEqual(len(set6), len(set7))
            
            set10 = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['compartment'], 
                compartName = ['e'], subSysName = None, multiCompartSub = 'e')
            self.assertGreater(len(set10), len(set1))

    
    def test_constrainRxnIndices(self):
        """ Check: all reactions of a type (e.g. "exchange") are set. Aggregate
            function only works on altered bounds, unless overrideBound is True. 
            Constrains are added on top of gene expression (with the assumption 
            that transport and exchange reactions are not bound by expression) """

        for mods in ['MODEL1507180055.xml']:
            modelRead = cobra.io.read_sbml_model('{0}/{1}'.format(modelDir, mods))
        
            exes = len(modelRead.exchanges)
            rxnInfo = gem.modelReactionInfo(modelRead)
            rxnIndices = gem.rxnIndexExtract(modelRead, rxnInfo, idType = 'exchanges')
            modelUpdate = gem.constrainRxnIndices(modelRead, rxnIndices, 
                indexLowerBound = -10, indexUpperBound = 10)
            self.assertEqual(exes, len(rxnIndices))
            self.assertEqual(len(rxnIndices), 
                len([x for x in modelUpdate.reactions if x.upper_bound == 10 or x.lower_bound == -10]))

            rxnIndices = gem.rxnIndexExtract(modelRead, rxnInfo, idType = ['exchanges', 
                'multicompartment'])
            modelUpdate = gem.constrainRxnIndices(modelRead, rxnIndices, 
                indexLowerBound = -10, indexUpperBound = 10)
            self.assertEqual(len(rxnIndices), 
                len([x for x in modelUpdate.reactions if x.upper_bound == 10 or x.lower_bound == -10]))
            
            
            modelUpdate.reactions[rxnIndices[0]].upper_bound = 20
            modelUpdate = gem.constrainRxnIndices(modelRead, rxnIndices, 
                indexLowerBound = -10, indexUpperBound = 10)
            self.assertEqual(modelUpdate.reactions[rxnIndices[0]].upper_bound, 15.)
            modelUpdate.reactions[rxnIndices[0]].upper_bound = 20
            modelUpdate = gem.constrainRxnIndices(modelRead, rxnIndices, 
                indexLowerBound = -10, indexUpperBound = 10, overrideBound = True)
            self.assertEqual(modelUpdate.reactions[rxnIndices[0]].upper_bound, 10.)
            modelUpdate = gem.constrainRxnIndices(modelRead, rxnIndices, 
                indexLowerBound = -10, indexUpperBound = 10, aggregateFun = 'min', 
                    overrideBound = False)
            self.assertEqual(modelUpdate.reactions[rxnIndices[0]].upper_bound, 10.)
            modelUpdate.reactions[rxnIndices[0]].upper_bound = 20
            modelUpdate = gem.constrainRxnIndices(modelRead, rxnIndices, 
                indexLowerBound = -10, indexUpperBound = 10, aggregateFun = 'mean', 
                    overrideBound = False, coef = 3)
            self.assertEqual(modelUpdate.reactions[rxnIndices[0]].upper_bound, 25.)
            modelUpdate.reactions[rxnIndices[0]].lower_bound = -20
            modelUpdate = gem.constrainRxnIndices(modelRead, rxnIndices, 
                indexLowerBound = -10, indexUpperBound = 10, aggregateFun = 'mean', 
                    overrideBound = False, coef = 1000)
            self.assertEqual(modelUpdate.reactions[rxnIndices[0]].lower_bound, -1000.)
        
            if '0055.xml' in mods:
                GEO = 'GSE59941'
            
                data, platform, organism = geo.experimentFilesReader(experimentName = GEO,
                    params = params)

                dataNorm = util.processData(expressionDF = data, params = params) 

                gammaShape, gammaLoc = geo.fitGamma(exprData = dataNorm, params = params)

                (modelRead, modelGenes, idMatch, geneDict,
                    _) = geo.readMetabolicModel(organism = organism, 
                        modelDict = modelDict, params = params)

                dataAgg, samples = geo.aggregateProbesToGenes(exprData = dataNorm, 
                    platform = platform, aggregateID = idMatch, params = params)

                dataModelMatch = geo.matchCBMGenes(sampleList = samples, 
                    modelGeneSet = modelGenes, exprDataWithIDs = dataAgg, geneID = idMatch,
                    gammaShape = gammaShape, gammaLoc = gammaLoc, params = params)

                samp = random.choice(samples)
                modelConstrain = geo.constrainModel(exprData = dataModelMatch, 
                    modelObject = modelRead, geneIDMatch = idMatch, geneDict = geneDict, 
                    sampleName = samp, params = params)
            
                origUnis = len(set(x.upper_bound for x in modelConstrain.reactions))
                modelConstrainUpdate = gem.constrainRxnIndices(modelConstrain, 
                    rxnIndices, indexLowerBound = -10, indexUpperBound = 10)
                self.assertLess(origUnis, len(set([x.upper_bound for x in modelConstrainUpdate.reactions])))
    
    def test_kcatAggregate(self):
        """ Check: kcat log conversion on nested dictionary. scaled values on
            nested dictionary ([0, 1000]). Aggregation of kcats, based on ECs, 
            pathways, or both. """

        with open('{0}/{1}'.format(modelDir, wormECMatchFile), 'r') as fin:
            rxnDict = json.load(fin)
        with open('{0}/{1}'.format(modelDir, kcatFile), 'r') as fin:
            kcatDict = json.load(fin)
        
        self.assertGreater(max(list(kcatDict['EC'].values())), 1000)
        rxnUpdate = gem.convertKCatReference(kcatDict, logKCat = True, scaleKCat = False)
        self.assertAlmostEqual(rxnUpdate['EC']['3.2.2.9'], 0.955511445)
        self.assertEqual(len(rxnUpdate['Reaction pathways']['KEGG']['00562']), 2)
        self.assertAlmostEqual(rxnUpdate['Reaction pathways']['KEGG']['00562'][0], 6.156978986)
        self.assertEqual(len(rxnUpdate['Reaction pathways']['KEGG']['00550']), 6)
        self.assertAlmostEqual(max(rxnUpdate['Reaction pathways']['KEGG']['00550']), 2.944438979) 

        rxnNoLog = gem.convertKCatReference(kcatDict, logKCat = False, scaleKCat = True)
        self.assertLessEqual(max(list(rxnNoLog['EC'].values())), 1000)

        with open('{0}/{1}'.format(modelDir, kcatFile), 'r') as fin:
            kcatDict = json.load(fin)

        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False, 
            matchECOnly = True, kcatAggregate = 'median')
        self.assertEqual(rxnDict['v59']['kcat'], 66.0)
        self.assertEqual(rxnDict['v323']['kcat'], 15.0)
        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False, 
            matchECOnly = True, kcatAggregate = 'max')
        self.assertEqual(rxnDict['v59']['kcat'], 66.0)
        self.assertEqual(rxnDict['v323']['kcat'], 39.0)

        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False, 
            matchPathwayOnly = True, choosePathway = False, kcatAggregate = 'median')
        self.assertEqual(rxnDict['v91']['kcat'], 17.0)
        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False, 
            matchPathwayOnly = True, choosePathway = False, kcatAggregate = 'max')
        self.assertEqual(rxnDict['v91']['kcat'], 1035.0)
        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False, 
            matchPathwayOnly = True, choosePathway = True, pathwayChoice = 'max',
            pathwayChoiceStat = 'median', kcatAggregate = 'min')
        self.assertEqual(rxnDict['v91']['kcat'], 37.0)
        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False, 
            matchPathwayOnly = True, choosePathway = True, pathwayChoice = 'min',
            pathwayChoiceStat = 'min', kcatAggregate = 'min')
        self.assertEqual(rxnDict['v91']['kcat'], 1.6)
        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False, 
            matchPathwayOnly = True, choosePathway = True, pathwayChoice = 'min',
            pathwayChoiceStat = 'median', kcatAggregate = 'min')
        self.assertEqual(rxnDict['v91']['kcat'], 6.7)

        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False,
            ECPreference = True)
        self.assertEqual(rxnDict['v99']['kcat'], 29.0)
        self.assertEqual(rxnDict['v190']['kcat'], 37.0)
        
        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = False, scaleKCat = False,
            ECPreference = True, choosePathway = True)
        self.assertEqual(rxnDict['v190']['kcat'], 385.0)

        rxnDict = gem.kcatAggregate(rxnDict, kcatDict, logKCat = True, scaleKCat = False,
            ECPreference = True, choosePathway = True)
        self.assertAlmostEqual(rxnDict['v190']['kcat'], 5.272683877)

        with open('{0}/{1}'.format(modelDir, mouseECMatchFile), 'r') as fin:
            rxnDictMou = json.load(fin)
        with open('{0}/{1}'.format(modelDir, kcatFile), 'r') as fin:
            kcatDict = json.load(fin)

        rxnDictMou = gem.kcatAggregate(rxnDictMou, kcatDict, logKCat = False, scaleKCat = False, 
            matchECOnly = True, kcatAggregate = 'median')
        self.assertEqual(rxnDictMou['v652']['kcat'], 654.0)
        self.assertEqual(rxnDictMou['v761']['kcat'], 31.5) #Multiple values

        modelReadMou = cobra.io.read_sbml_model('{0}/MODEL1507180055.xml'.format(modelDir))
        modelConstrainMou = gem.constrainByKCat(modelReadMou, rxnDictMou, overrideBound = True,
            finishScale = False)
        self.assertEqual(modelConstrainMou.reactions[2531].upper_bound, 1.5)
        modelReadMou = cobra.io.read_sbml_model('{0}/MODEL1507180055.xml'.format(modelDir))
        modelReadMou = geo.prepModelConstraints(modelReadMou, 
            lowerBound = -1000, upperBound = 1000)
        modelReadMou.reactions[2531].upper_bound = 999.99999
        modelConstrainMou = gem.constrainByKCat(modelReadMou, rxnDictMou, overrideBound = False,
            prepConstraints = False, finishScale = False)
        self.assertAlmostEqual(modelConstrainMou.reactions[2531].upper_bound, 500.749995)
       
if __name__ == '__main__':

    unittest.main()