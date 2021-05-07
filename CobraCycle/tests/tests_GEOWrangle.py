""" Note: given how long it can take to read metabolic models and aggreagte across
        tens of thousands of rows, unit-testing will be performed by "modules" of
        the pipeline, rather than each individual function. In doing so, the earlier
        steps of the pipeline do not need to be re-run repeatedly, and checks are
        instead made "along the way". """

import unittest, sys, os, re, warnings, cobra, random, json, yaml
sys.path.append('CobraCycle/src/')
import setup_GEOWrangle as filePull
import setup_DataNormalization as dataProcess
import numpy as np
from pandas.util.testing import assert_frame_equal
from itertools import chain

paramFile = 'CobraCycle/params/GEO_GeneExprProcess.yaml'

with open(paramFile) as fin:
    params = yaml.safe_load(fin)

with open(params['modelDictFile']) as fin:
    modelDict = json.load(fin)

class TestDownloads(unittest.TestCase):


    def test_downloads_organismID_Normalize_expandID_HGNC(self): #"Module 2"
        """ Check: non-existant GSE gives correct exception. Lack of annotation file
            gives correct exception. Proper GSE study can be succcessfully queried
            with the right organismID. Normalized dataset throws exception. 
            Columns are the same before and after expandPlatformIDs(). New object
            len after expandPlatformIDs() is g.t.e the original. A bad separator
            returns the same object. For human studies, if HGNC file isn't present
            (or misspecified), the HGNC json is queried. If expandPlatform is True,
            the len of the result is g.t.e expandPlatform = False. If sep is 
            something else, the len of the results equals expandPlatform = False """
        
        geoDict = {'GSE19109' : {'Organism': 'Arabidopsis thaliana'},
            'GSE37307': {'Organism': 'Homo sapiens'},
            'GSE54536': {'Organism': 'Homo sapiens'},
            'GSE59941': {'Organism': 'Mus musculus'},
            'GSE71117': {'Organism': 'Gallus gallus'},
            'GSE9785': {'Organism': 'Mus musculus'},
            'GSE8708': {'Organism': 'Escherichia coli'},
            'GSE21784': {'Organism': 'Caenorhabditis elegans'},
            'GSE7614': {'Organism': 'Drosophila melanogaster'},
            'GSE15374': {'Organism': 'Gycine max'}}

        with self.assertRaises(Exception) as msg: 
           filePull.downloadExpFiles('GSE999999')
        the_except = msg.exception
        self.assertEqual(str(the_except), 
            'No .txt matrix file present for this experiment')
        with self.assertRaises(Exception) as msg1: 
           filePull.downloadExpFiles('GSE99999')
        the_except1 = msg1.exception
        self.assertEqual(str(the_except1), 
            'Possible RNASeq experiment detected')
        
        geo = np.random.choice(list(geoDict.keys()))
        if geo == 'GSE15374':
            with self.assertRaises(Exception) as msg2: 
                filePull.experimentFilesReader(geo)
            the_except2 = msg2.exception
            self.assertEqual(str(the_except2),
                'Data appears roughly normally distributed and was likely centered')
        else:
            data, platform, organism = filePull.experimentFilesReader(geo)
            self.assertIsNotNone(data)
            self.assertIsNotNone(platform)
            self.assertEqual(organism, geoDict[geo]['Organism'])
            
            validCols = [ID for ID in platform.columns if 
                len(set(platform[ID][platform[ID].notnull()]))>0]
            colName = np.random.choice(validCols)
            if any(platform[colName].astype(str).str.contains('///')):
                platformExpand = filePull.expandPlatformIDs(platform, colName, '///')
                self.assertLess(len(platform), len(platformExpand))
                self.assertCountEqual(list(platform.columns), list(platformExpand.columns))
            else:
                platformExpandNeg = filePull.expandPlatformIDs(platform, colName, '///')
                self.assertEqual(platform.shape, platformExpandNeg.shape)
            platformNeg = filePull.expandPlatformIDs(platform, colName, '###')
            self.assertEqual(platform.shape, platformNeg.shape)
    
        
    
    def test_findMaxID_aggProbesGenes_matchCBM_constrain_reset(self): #"Module 3"
        """ Check: Output of findMaxID is a string of a column from platform.columns.
            checkSeparation=True returns the same column and that the number of 
            matches is at least that with checkSeparation=False (outside the function).
            For aggregateProbes(), max or median aggregation works. Sample name 
            extraction is correct. Calling expandID within the function vs in 
            expandPlatformIDs() gives the same result. New object has len l.t. 
            original data. If re-joining on the platform results in more rows than
            unique genes, check that the expression values are the same for each 
            duplicate gene ID. Matching to metabolic model in matchCBM() returns 
            data frame of len()>0. If no ID matches are found, then findMaxIDOnFailure
            finds the gene ID match and still returns a valid data frame. Check
            unique set of bounds increases before and after constrain, and the set 
            of bounds contains only 0, 1000, or a value from the sample used for
            constraining. Different aggregations for constrainModel() are possible.
            The number of unique flux bounds is the same as the original after 
            applying resetModelBounds(). """
        
        preferredWorm = np.random.choice(['WormFlux'])
        modelPath = 'ref'
        geoList = [modelDict[x]['ExampleGSE'] for x in modelDict]
        geo = np.random.choice(geoList)
        data, platform, organism = filePull.experimentFilesReader(geo)

        modelRead, modelGenes, geneID, geneDict, boundDict = filePull.readMetabolicModel(organism, 
                modelDict, modelPath = 'ref', preferredWorm = preferredWorm)
        
        idMatchF = filePull.findMaxIDMatch(platform, modelGenes, 
            checkSeparation = False, sep = '///')
        self.assertIn(idMatchF, platform.columns)
        idMatchT = filePull.findMaxIDMatch(platform, modelGenes, 
            checkSeparation = True, sep = '///')
        self.assertIn(idMatchT, platform.columns)
        self.assertEqual(idMatchF, idMatchT)
        self.assertEqual(geneID, idMatchT)
        uniValsF = set(platform[idMatchF])
        lenF = len(modelGenes.intersection(uniValsF))
        uniValsT = set(chain.from_iterable([re.split('///', 
            str(x)) for x in list(platform[idMatchT])]))
        lenT = len(modelGenes.intersection(uniValsT))
        self.assertLessEqual(lenF, lenT)
    
        dataNorm = dataProcess.processData(data)
        gammaShape, gammaLoc = filePull.fitGamma(dataNorm, **params)

        validCols = [ID for ID in platform.columns if 
                len(set(platform[ID][platform[ID].notnull()]))>0]
        colName = np.random.choice(validCols)
        dataAgg1, samples1 = filePull.aggregateProbesToGenes(dataNorm, platform, 
            aggregateID = colName, expandID = True, sep = '///', 
            aggregateFun = 'max')
        dataAgg2, _ = filePull.aggregateProbesToGenes(dataNorm, platform, 
            aggregateID = colName, expandID = True, sep = '///',
            aggregateFun = 'median')
        self.assertEqual(dataAgg1.shape, dataAgg2.shape)
        self.assertCountEqual(list(samples1), list(data.columns[1:]))

        #Lines within the aggreagteProbesToGenes() function to re-create
        platformTrim = platform[platform[colName].notnull()]
        if any(platformTrim[colName].astype(str).str.contains('///')):   
            platformExpand = filePull.expandPlatformIDs(platformTrim, colName, '///')
            dataAgg3, _ = filePull.aggregateProbesToGenes(dataNorm, platformExpand, 
                aggregateID = colName, expandID = False, sep = '///', 
                aggregateFun = 'max')
            self.assertEqual(dataAgg1.shape, dataAgg3.shape)
        
        randomGenes = np.random.randint(0, len(set(dataAgg1[colName])), 100)
        geneList = list(set(dataAgg1[colName]))
        for gene in randomGenes:
            oneGene = dataAgg1[samples1][dataAgg1[colName]==geneList[gene]]
            if len(oneGene) > 1:
                uniVals = set(np.ravel(oneGene))
                self.assertLessEqual(len(uniVals), len(oneGene.columns))
        
        dataAggPos, samplesPos = filePull.aggregateProbesToGenes(dataNorm, platform, 
            aggregateID = geneID, expandID = True, sep = '///', 
            aggregateFun = 'max')
        
        randomGenes2 = np.random.choice(dataAggPos[geneID], 5)
        randomSample2 = np.random.choice(samplesPos)

        dataModelMatch = filePull.matchCBMGenes(samplesPos, modelGenes, dataAggPos,
            geneID, gammaShape, gammaLoc, scaleVals = True, 
            minFluxVal = 0, maxFluxVal = 1000, findMaxIDOnFailure = False, 
            checkSeparation = True, sep = '///', sampleMissingGenes = True)
        self.assertGreater(len(dataModelMatch), 0)
        self.assertGreaterEqual(len(dataModelMatch), len(modelGenes))

        for gene in randomGenes2:
            if gene in dataModelMatch[geneID]:
                preMatch = dataAggPos[randomSample2][dataAggPos[geneID] == gene].values[0]
                postMatch = dataModelMatch[randomSample2][dataModelMatch[geneID] == gene].values[0]
                self.assertEqual(preMatch, postMatch)
        
        nullCol = np.random.choice([x for x in platform.columns if x not in validCols])
        dataModelMatch2 = filePull.matchCBMGenes(samplesPos, modelGenes, dataAggPos,
            nullCol, gammaShape, gammaLoc, scaleVals = True, 
            minFluxVal = 0, maxFluxVal = 1000, findMaxIDOnFailure = True, 
            checkSeparation = True, sep = '///', sampleMissingGenes = False)
        self.assertGreater(len(dataModelMatch2), 0)

        sampleName = np.random.choice([x for x in dataModelMatch.columns if 'GSM' in x])
        exprVals = set(dataModelMatch[sampleName])
    
        preBounds = set([rxn.upper_bound for rxn in modelRead.reactions])
        modelRead = filePull.prepModelConstraints(modelRead, -1000, 1000)
        testUppers = [rxn.upper_bound for rxn in modelRead.reactions]
        testLowers = [rxn.lower_bound for rxn in modelRead.reactions]
        self.assertFalse(any([x > 1000 for x in testUppers]))
        self.assertFalse(any([x < -1000 for x in testLowers]))

        modelConstrainPos = filePull.constrainModel(exprData = dataModelMatch, 
            modelObject = modelRead, geneIDMatch = geneID, geneDict = geneDict,
            sampleName = sampleName, prepConstraints = True, 
            lowerBound =  -1000, upperBound = 1000, constrainFun = 'max')
        postBounds = set([rxn.upper_bound for rxn in modelConstrainPos.reactions])
        self.assertGreater(len(postBounds), len(preBounds))
        self.assertGreater(len(exprVals.intersection(postBounds)), 0)
        #self.assertNotEqual(len(exprVals.intersection(postBounds)), len(postBounds))
        
        modelReadAlt, _, geneID, geneDict, _ = filePull.readMetabolicModel(organism, 
            modelDict, modelPath = 'ModelRepository', preferredWorm = preferredWorm)
        modelConstrainAlt = filePull.constrainModel(exprData = dataModelMatch, 
            modelObject = modelReadAlt, geneIDMatch = geneID, geneDict = geneDict,
            sampleName = sampleName, prepConstraints = True,
            lowerBound =  -1000, upperBound = 1000, constrainFun = 'median')
        postBoundsAlt = set([rxn.upper_bound for rxn in modelConstrainAlt.reactions])
        self.assertGreater(len(postBoundsAlt), len(preBounds))

        modelReset = filePull.resetModelBounds(modelConstrainPos, boundDict)
        resetBounds = set([rxn.upper_bound for rxn in modelReset.reactions])
        self.assertEqual(len(preBounds), len(resetBounds))
        

    def test_extractGEOSampleParams(self):
        """ Check: A random sample from a GEO can be queried for the matching GEO, 
            correct organism, and detection of RNA (and expression set to True).
            Characteristics for c elegans study is a dict and not str.
            Added structured field from the GEO url page can be added """

        with self.assertRaises(ValueError): 
           filePull.extractGEOSampleParams('GSM000000')

        geoList = [modelDict[x]['ExampleGSE'] for x in modelDict]
        geo = np.random.choice(geoList)
        data, _, organism = filePull.experimentFilesReader(geo)
        samples = [x for x in data.columns if 'GSM' in x]
        samp = random.choice(samples)
        expParams = filePull.extractGEOSampleParams(sampleID = samp, 
            parseIDs = ['Title', 'Source name', 'Characteristics',
                         'Sample type', 'Organism', 'Extracted molecule'])
        self.assertEqual(expParams['GSE'], geo)
        self.assertEqual(expParams['Expression'], True)
        if organism == 'Caenorhabditis elegans':
            self.assertTrue(isinstance(expParams['Characteristics'], dict))
        self.assertTrue('RNA' in expParams['Extracted molecule'])


    def test_prepEPDict(self):
        """ Check: 'Data' key under the returned dict contains all entries of 
            len == nRxns. The lb and ub arrays equal the modelObject. 
            The storeEPDir is correctly specified under the 'MetaData' key.
            For GEOtoEP, the desired .json is in the right directory, and the 
            epDict() is removed from the main MetabolicEP directory """
        model = np.random.choice(list(modelDict.keys()))
        geo = modelDict[model]['ExampleGSE']
            
        data, platform, organism = filePull.experimentFilesReader(experimentName = geo,
            **params)

        dataNorm = dataProcess.processData(expressionDF = data, **params) 

        gammaShape, gammaLoc = filePull.fitGamma(exprData = dataNorm, **params)

        (modelRead, modelGenes, idMatch, geneDict,
            boundDict) = filePull.readMetabolicModel(organism = organism, 
                modelDict = modelDict, **params)

        dataAgg, samples = filePull.aggregateProbesToGenes(exprData = dataNorm, 
            platform = platform, aggregateID = idMatch, **params)

        dataModelMatch = filePull.matchCBMGenes(sampleList = samples, 
            modelGeneSet = modelGenes, exprDataWithIDs = dataAgg, geneID = idMatch,
            gammaShape = gammaShape, gammaLoc = gammaLoc, **params)

        samp = random.choice(samples)
        modelConstrain = filePull.constrainModel(exprData = dataModelMatch, 
            modelObject = modelRead, geneIDMatch = idMatch, geneDict = geneDict, 
            sampleName = samp, **params)

        epInfoDict = filePull.prepEPDict(modelObject = modelConstrain, 
            sampleID = samp, modelName = model, **params, parseIDs = ['Title', 
            'Source name', 'Characteristics', 'Sample type', 'Organism'])

        trueLBs = [rxn.lower_bound for rxn in modelConstrain.reactions]
        trueUBs = [rxn.upper_bound for rxn in modelConstrain.reactions]
        testLBs = []
        testUBs = []
        for rxn in epInfoDict['Data']:
            testLBs.append(epInfoDict['Data'][rxn]['lb'])
            testUBs.append(epInfoDict['Data'][rxn]['ub'])
        self.assertEqual(testLBs, trueLBs)
        self.assertEqual(testUBs, trueUBs)
        self.assertEqual(params['storeEPDir'], epInfoDict['MetaData']['Store'])

        if model in ['WormFlux', 'Mouse']:
            filePull.executeGEOtoEP(epDirectory = 'MetabolicEP', 
                    epDict = epInfoDict, handleScript = 'GEOtoEPFit.py', 
                    pyVersion = '3.6', defaultYAML = True, removeDict = False)
            epTempFile = epInfoDict['MetaData']['tag'] + '_epTemp.json'
            self.assertTrue(epTempFile in os.listdir('../MetabolicEP'))
            os.remove('../MetabolicEP/{0}'.format(epTempFile))
            modelFolder = re.sub(r'\ |\.', '', model)
            resultFile = '{0}_storeEP.json'.format(samp)
            self.assertTrue(resultFile in os.listdir('../MetabolicEP/resultStore/{0}/{1}'.format(modelFolder, 
                geo)))
            os.remove('../MetabolicEP/resultStore/{0}/{1}/{2}'.format(modelFolder, 
                geo, resultFile))
         
        if model == 'WormFlux':
            modelRead = filePull.resetModelBounds(modelObject = modelRead, 
                boundDict = boundDict)
            samp = 'GSM542652'

            modelConstrain = filePull.constrainModel(exprData = dataModelMatch, 
            modelObject = modelRead, geneIDMatch = idMatch, geneDict = geneDict, 
            sampleName = samp, **params)

            epInfoDict = filePull.prepEPDict(modelObject = modelConstrain, 
                sampleID = samp, modelName = model, **params, parseIDs = ['Title', 
                'Source name', 'Characteristics', 'Sample type', 'Organism'])

            filePull.executeGEOtoEP(epDirectory = 'MetabolicEP', 
                epDict = epInfoDict, handleScript = 'GEOtoEPFit.py', 
                pyVersion = '3.9', defaultYAML = True, removeDict = True)

            with open('../MetabolicEP/resultStore/WormFlux/{0}/{1}_storeEP.json', 'r') as fin:
                output = json.load(fin)

            with open('../MetabolicEP/resultStore/WormFlux/{0}_Control_storeEP.json', 'r') as fin:
                posControl = json.load(fin)
            
            outputLoc = output['conLoc']
            controlLoc = posControl['conLoc']
            self.assertTrue(outputLoc, controlLoc)

if __name__ == '__main__':

    for f in os.listdir():
        if re.search('abc.gz', f):
            os.remove(f)

    geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708']
    for geo in geoList:
        if not any([re.search(geo, x) for x in os.listdir(params['geoDir'])]):
            readIn, _, _ = filePull.experimentFilesReader(geo)

    unittest.main()

