""" With a list of GSE studies, query, process, and fit EP to each sample

    Functionalities of this script:
        - Device placement - Assuming 0 or 1 as options, and the GEO list will
                                be split accordingly - check
        - Checks for existing results, so as not to run multiple times
        - Catch exceptions and throw into a dict counter, to track how often and
            why functions fail (GEO and Sample specific) - check

    Dec 21, 2018
    Seth Rhoades
"""
import csv, json, yaml, sys, os, time, re, warnings, subprocess
sys.path.append('./src/')
import setup_GEOWrangle as geoUtil 
import setup_DataNormalization as dataUtil
import setup_gemConstrain as gemUtil
Device = int(sys.argv[1])

WormQuery = 'ref/Celegans_GSE_GEOQuery.txt' 
MouseQuery = 'ref/MMusculus_GSE_GEOQuery.txt'

wormGEOs = geoUtil.queryListGEOQuery(WormQuery, organism = 'Caenorhabditis elegans',
    geoFilter = 'GSE', studyType = 'expression')
mouseGEOs = geoUtil.queryListGEOQuery(MouseQuery, organism = 'Mus musculus',
    geoFilter = 'GSE', studyType = 'expression')
comboGEOs = wormGEOs + mouseGEOs

if Device == 0:
    comboGEOs = comboGEOs[:int(len(comboGEOs) / 2)]
if Device == 1:
    comboGEOs = comboGEOs[int(len(comboGEOs) / 2):]

paramFile = 'params/GEO_GeneExprProcess.yaml'
with open(paramFile) as fin:
    params = yaml.safe_load(fin)
with open(params['modelDictFile']) as fin:
    modelDict = json.load(fin)

def main(params, comboGEOs, modelDict, Device):
    
    geoError = dict()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')

        for geo in comboGEOs:
            start_time = time.time()

            if geo in wormGEOs:
                organism = 'Caenorhabditis elegans'
                model = 'WormFlux'
                if params['addKCatConstrain'] == True:
                    with open('{0}/{1}'.format(params['modelPath'], 
                        params['wormECMatchFile']), 'r') as fin:
                        rxnECPathwayDict = json.load(fin)
                    
            if geo in mouseGEOs:
                organism = 'Mus musculus'
                model = 'Mouse'
                if params['addKCatConstrain'] == True:
                    with open('{0}/{1}'.format(params['modelPath'], 
                        params['mouseECMatchFile']), 'r') as fin:
                        rxnECPathwayDict = json.load(fin)

            if params['addKCatConstrain'] == True:
                with open('{0}/{1}'.format(params['modelPath'], 
                    params['kcatFile']), 'r') as fin:
                        kcatDict = json.load(fin)
            
            rxnECPathwayDict = gemUtil.kcatAggregate(rxnECPathwayDict, kcatDict, 
                **params)
        
            if geo not in os.listdir('../MetabolicEP/resultStore/{0}'.format(model)):
                print('Processing GSE study {0}'.format(geo))
                keepGoing = False
                geoError[geo] = dict()
                geoError[geo]['Organism'] = organism
                
                try:

                    data, platform, organism = geoUtil.experimentFilesReader(experimentName = geo,
                        **params)

                    dataNorm = dataUtil.processData(expressionDF = data, **params) 

                    gammaShape, gammaLoc = geoUtil.fitGamma(exprData = dataNorm, **params)

                    (modelRead, modelGenes, idMatch, geneDict,
                        boundDict) = geoUtil.readMetabolicModel(organism = organism, 
                            modelDict = modelDict, **params)

                    rxnInfo = gemUtil.modelReactionInfo(modelRead)
                    rxnIndices = gemUtil.rxnIndexExtract(modelRead, rxnInfo, 
                        idType = params['idType'])
                    
                    dataAgg, samples = geoUtil.aggregateProbesToGenes(exprData = dataNorm, 
                        platform = platform, aggregateID = idMatch, **params)

                    dataModelMatch = geoUtil.matchCBMGenes(sampleList = samples, 
                        modelGeneSet = modelGenes, exprDataWithIDs = dataAgg, geneID = idMatch,
                        gammaShape = gammaShape, gammaLoc = gammaLoc, **params)
                    
                    geoError[geo]['Error'] = 'None'
                    geoError[geo]['Samples'] = dict()
                    keepGoing = True
                
                except Exception as err:
                    geoError[geo]['Error'] = str(err)

                if keepGoing == True:
                    for samp in samples:
                        
                        try:

                            modelConstrain = geoUtil.constrainModel(exprData = dataModelMatch, 
                                modelObject = modelRead, geneIDMatch = idMatch, 
                                geneDict = geneDict, sampleName = samp, **params)

                            if params['addConstraints'] == True:
                                modelConstrain = gemUtil.constrainRxnIndices(modelConstrain, 
                                    rxnIndices, **params)
                            if params['addKCatConstrain'] == True:
                                modelConstrain = gemUtil.constrainByKCat(modelConstrain,
                                    rxnECPathwayDict, **params)
                            
                            epInfoDict = geoUtil.prepEPDict(modelObject = modelConstrain, 
                                sampleID = samp, modelName = model, 
                                parseIDs = ['Title', 'Source name', 'Characteristics', 
                                'Sample type', 'Organism'], nDevices = 2, 
                                specifyDevice = Device, **params)

                            geoUtil.executeGEOtoEP(epDirectory = 'MetabolicEP', 
                                epDict = epInfoDict, handleScript = 'GEOtoEPFit.py', 
                                pyVersion = '3.9', removeDict = True)
                            
                            modelRead = geoUtil.resetModelBounds(modelObject = modelRead, 
                                boundDict = boundDict)
                            
                            geoError[geo]['Samples'][samp] = 'Success'
                            

                        except Exception as err:
                            geoError[geo]['Samples'][samp] = str(err)
                    
                    for f in os.listdir(params['geoDir']):
                        if re.search(geo, f):
                            os.remove('{0}/{1}'.format(params['geoDir'], f))
            
            print("--- {0} seconds for study {1} ---".format(time.time() - start_time, geo))

    print('GEO list done')
    with open('GEOtoEP_queryResultSuccess_Device{0}.json'.format(Device), 'w') as fout:
        json.dump(geoError, fout, indent = 4)
        fout.write('\n')
    
if __name__ == '__main__':

    main(params, comboGEOs, modelDict, Device)
    
