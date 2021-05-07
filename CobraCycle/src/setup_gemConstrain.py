""" Added GEM model constraints """

import setup_GEOWrangle as geo
import numpy as np
from sklearn.preprocessing import MinMaxScaler

compartDict = {'MODEL1604210000': {'c': 'Cytosol', 'm': 'Mitochondria', 
                                'e': 'Extracellular',
                                'cytosol': 'c', 'cytoplasm': 'c', 
                                'extracellular': 'e',
                                'mitochondria': 'm', 'mitochondrion': 'm'},
                'MODEL1507180055': {'cytoplasm': 'c', 'cytosol': 'c', 
                                    'extracellular': 'e', 'golgi': 'g', 'golgi apparatus': 'g', 
                                    'lysosome': 'l', 'mitochondria': 'm', 'mitochondrion': 'm', 
                                    'nucleus': 'n', 'endoplasmic reticulum': 'r', 'er': 'r', 
                                    'glyoxysome': 'x'},
                }
     

def modelReactionInfo(modelObject, extracts = ['subsystem', 'compartments'],
    convertWorm = True):
    """ Return a dictionary of reactions (by index) from the metabolic model which
        contain information such as their compartment of subsystem/pathway. 
        Extract must contain items (case-sensitive) in dir(modelObject.reactions).
        Note, not all models contain subsystem information. Additional keys are
        added for reactions which contain multiple compartments (transport), and 
        reactions round in the modelObject.exchanges. Note Wormflux has unique
        compartment-naming mechanism.

        General format:
            - ['Genes']
                - ['Yes'] = [1, 2, 4]
                - ['No'] = [3, 5]
            - ['Subsystem']
                - ['Glycolysis'] = [1, 2]
                - ['Transport'] = [3, 4, 5]
            - ['Compartment']
                - ['Cytosol'] = [1, 2, 3, 4]
                - ['Mitochondria'] = [5]
    """        

    if any([x for x in extracts if x not in dir(modelObject.reactions[0])]):
        raise ValueError("Items in {0} are not found in model's reaction "
            "items".format(extracts))

    rxnDict = dict()
    for ex in extracts:
        if ex != 'name':
            rxnDict[ex] = dict()
        else:
            rxnDict[ex] = []
    if 'genes' in extracts:
        rxnDict['genes']['yes'] = []
        rxnDict['genes']['no'] = []
    
    rxnDict['exchanges'] = []
    rxnDict['multicompartment'] = []
    exchangeIDs = [x.id for x in modelObject.exchanges]
 
    for i, rxn in enumerate(modelObject.reactions): #Prep the nest
        for ex in extracts:
            
            if ex == 'genes':
                if len(rxn.genes) > 0:
                    rxnDict[ex]['yes'].append(i)
                else:
                    rxnDict[ex]['no'].append(i)
            
            elif ex == 'name':
                rxnDict[ex].append(rxn.name)

            else:
                val = ''.join(list(rxn.__getattribute__(ex)))
                if val not in rxnDict[ex]:
                    rxnDict[ex][val] = []
                rxnDict[ex][val].append(i)
        
        if rxn.id in exchangeIDs:
            rxnDict['exchanges'].append(i)
        if len(rxn.compartments) > 1:
            rxnDict['multicompartment'].append(i)
    
    if (convertWorm == True and modelObject.id == 'iCEL1273' and 
        'compartments' in rxnDict):
        replaceDict = dict()
        for c in rxnDict['compartments']:
            replaceDict[c[0].lower()] = rxnDict['compartments'][c]
        rxnDict['compartments'] = replaceDict

    for i in rxnDict:
        if len(rxnDict[i]) == 0:
            print('Warning: {0} item has no entries'.format(i))
    if len(set(rxnDict['exchanges']).intersection(rxnDict['multicompartment'])) > 0:
        print('Warning. Reactions appear as both exchanges and transports')
    if len(rxnDict['exchanges']) == 0 or len(rxnDict['multicompartment']) == 0:
        raise Exception('No exchange or transport reactions found')

    return rxnDict


def rxnIndexExtract(modelObject, rxnInfo, idType, compartName = None, 
    subSysName = None, multiCompartSub = None, **params):
    """ Extract reaction indices from modelReactionInfo() ouput. Flexibile inputs
        to extract any of subsystem(s), compartment(s), or all 'exchange' and  
        'multicompartment'. Each model has their own compartment naming mechanism,
        so a best attempt str match will be made. This list of indices will be 
        fed to a downstream model constraint function. For compartments, if
        compartName is None and multiCompartSub is specified, then all 
        multicomponent reactions which contain multiCompartSub will be returned.
        If compartName and multiCompartSub are specified, then the compartName(s)
        will be returned, along with any multicompartments which contain
        the multiCompartSub (e.g. return me any/all reactions which are only
        extracellular or extracellular <-> cytosol).

        Args:
            modelObject: GEM model - Model object, for id extraction
            rxnInfo: Dict - from modelReactionInfo()
            idType: Str/List - Type of match(es), must be in rxnInfo
            compartName: Str/list - Name of compartment (if idType == 'compartments')
                Flexible, and also can be used to extract multicompartments.
            subSysName: Str/list - Name of subsystem. Flexible for one or 
                multiple
            multiCompartSub: Str/list - Extract a subset of multicompartments
                which contain the item(s) in multiCompartSub (e.g. only 'e' for
                    extracellular)
            returnEmpty: Bool - If improper input, or no matches found, return
                an empty list

        Returns:
            rxnIndicies: List - List of rxn indices in modelObject
    """
    modID = modelObject.id 
    if modID not in compartDict:
        raise Exception('Model {0} not found in compartment translation '
            'dict'.format(modID))

    if isinstance(idType, str):
        idType = [idType]

    idTypeLow = [x.lower() for x in idType]
    for item in range(len(idTypeLow)):
        if idTypeLow[item] in ['exchange', 'compartment']:
            idTypeLow[item] = idTypeLow[item] + 's'

    if not any([x for x in idTypeLow if x in rxnInfo]):
        raise ValueError('No specified id types found in the modelReactionInfo() '
                'input. Choose among {1}'.format(list(rxnInfo.keys())))
    
    if compartName is not None:
        if isinstance(compartName, str):
            compartName = [compartName]
        compartNameLow = [x.lower() for x in compartName]
    if multiCompartSub is not None:
        if isinstance(multiCompartSub, str):
            multiCompartSub = [multiCompartSub]
        multiCompartSubLow = [x.lower() for x in multiCompartSub]
    if subSysName is not None:
        if isinstance(subSysName, str):
            subSysName = [subSysName]
        subSysNameLow = [x.lower() for x in subSysName]

    if compartName is not None:
        for c in range(len(compartNameLow)):
            if (compartNameLow[c] not in rxnInfo['compartments'] and 
                compartName[c] not in rxnInfo['compartments']):
                
                if (len(compartName[c]) > 1 and 
                    compartNameLow[c] not in compartDict[modID]):
                    print('Warning, {0} may not be a valid compartment. Try to '
                        'enter either a one-letter abbrevation, or the full '
                        'compartment name (e.g. "cytosol")'.format(compartName[c]))

                if compartNameLow[c] in compartDict[modID]:
                    compartNameLow[c] = compartDict[modID][compartNameLow[c]]
                elif compartName[c] in compartDict[modID]:
                    compartName[c] = compartDict[modID][compartName[c]]
    
    if multiCompartSub is not None:
        for c in range(len(multiCompartSubLow)):
            if (multiCompartSubLow[c] not in rxnInfo['compartments'] and 
                multiCompartSub[c] not in rxnInfo['compartments']):

                if (len(multiCompartSub[c]) > 1 and 
                    multiCompartSubLow[c] not in compartDict[modID]):
                    print('Warning, even though you are checking for partial '
                        'matches for multicompartment reactions, {0} may not be '
                        'a valid compartment. Try to enter either a one-letter '
                        'abbrevation, or the full compartment name '
                        '(e.g. "cytosol")'.format(multiCompartSub[c]))

                if multiCompartSubLow[c] in compartDict[modID]:
                    multiCompartSubLow[c] = compartDict[modID][multiCompartSubLow[c]]
                elif multiCompartSub[c] in compartDict[modID]:
                    multiCompartSub[c] = compartDict[modID][multiCompartSub[c]]
    
    rxnIndices = []
    for item in idTypeLow:
        if item == 'compartments':
            compartIndices = []
            if compartName is None and multiCompartSub is None:
                print('Compartment type specified, but no compartment names given')
            
            if compartName is not None:
                for cUser in range(len(compartNameLow)):
                    for cBase in rxnInfo['compartments']:
                        if compartNameLow[cUser] == cBase:
                            compartIndices += rxnInfo['compartments'][cBase]
                        elif compartName[cUser] == cBase:
                            compartIndices += rxnInfo['compartments'][cBase]

            if multiCompartSub is not None:
                for cUser in range(len(multiCompartSubLow)):
                    for cBase in rxnInfo['compartments']:
                        if multiCompartSubLow[cUser] in cBase: #Compartments should alreay be lower
                            compartIndices += rxnInfo['compartments'][cBase]
                        elif multiCompartSub[cUser] in cBase:
                            compartIndices += rxnInfo['compartments'][cBase]

            rxnIndices += compartIndices

        if item == 'subsystem':
            subSysIndices = []
            if subSysName is None:
                print('Subsystem type specified, but no subsystem names given')

            else:
                for cUser in range(len(subSysNameLow)):
                    for cBase in rxnInfo['subsystem']:
                        if subSysNameLow[cUser] == cBase.lower(): #Subsystems could be any case
                            subSysIndices += rxnInfo['subsystem'][cBase]
                        elif subSysName[cUser] == cBase:
                            subSysIndices += rxnInfo['subsystem'][cBase]

            rxnIndices += subSysIndices
        
        if item == 'exchanges':
            rxnIndices += rxnInfo['exchanges']

        if item == 'multicompartment':
            rxnIndices += rxnInfo['multicompartment']

    rxnIndices = sorted(list(set(rxnIndices)))
    if len(rxnIndices) == 0:
        print('No reactions were found. Check your inputs (do your compartments '
            'or subsystems exist in the model object?)')

    return rxnIndices


def constrainRxnIndices(modelObject, rxnIndices, indexLowerBound = -10, 
    indexUpperBound = 10, aggregateFun = 'mean', coef = 1., 
    prepConstraints = True, overrideBound = False, finishScale = True, 
    allLowerBound = -1000, allUpperBound = 1000, **params):
    """ Take a set of reaction indicies and constrain. Optionally, check the
        original model bounds before constraining (and only constrain if the
        current model bounds are not equivalent). Or constrain anyways with an
        override argument. If the reaction is already constrained, combine values
        through some aggregation. A weight can be given to the reaction index
        constraint. The constraint is checked by comparing flux values to the
        allLower and allUpper bounds. If other constraints had been used, e.g. 
        from gene expession, these values are protected, even if 
        prepModelConstraints() is executed.
    
    Args:
        modelObject: GEM object
        rxnIndicies: List - Reactions, by index in modelObject.reactions
        indexLowerBound: Float - Lower bound on rxnIndices
        indexUpperBound: Float - Upper bound on rxnIndices
        aggregateFun: Str - aggregation function ('min', 'max', 'mean')
        coef: Float - If reaction is already constrained, weight coefficient
            of the indexLower or indexUpperBounds
        prepConstraints: Bool - Prep model constraints first
        overrideBound: Bool - Override an existing bound, and dont aggregate
        finishScale: Bool - Re-scale all values within allLower and allUpperBounds
            after constraints
        allLowerBound: Float - Lower bound for prepConstraints
        allUpperBound: Float - Upper bound for prepConstraints

    Returns:
        modelObject
    """
    if (not isinstance(indexLowerBound, float) and 
        not isinstance(indexLowerBound, int) or 
        not isinstance(indexUpperBound, float) and 
        not isinstance(indexUpperBound, int) or
        not isinstance(coef, float) and 
        not isinstance(coef, int)):
        raise ValueError('Bounds and coef need to be int or float')
    
    if not isinstance(rxnIndices, list):
        raise ValueError('Reaction indices must be a list')

    if indexLowerBound > 0:
        raise ValueError('Lower bound must be less l.t.e 0')
    if indexUpperBound < 0:
        raise ValueError('Upper bound must be g.t.e 0')

    if aggregateFun not in ['min', 'max', 'mean']:
        raise ValueError('Aggregation function must be "min", "max", or "mean"')

    origUnis = len(set([x.upper_bound for x in modelObject.reactions]))

    if prepConstraints == True:
        modelObject = geo.prepModelConstraints(modelObject, 
            lowerBound = allLowerBound, upperBound = allUpperBound)
    
    rxnLen = len(modelObject.reactions)
    if len(set(rxnIndices).intersection(range(rxnLen))) != len(rxnIndices):
        raise Exception('Reaction indicies contains values outside the range '
            'of reactions in the metabolite model')

    for i, _ in enumerate(modelObject.reactions):
        if i in rxnIndices:
            if (modelObject.reactions[i].lower_bound != 0 and 
                modelObject.reactions[i].lower_bound != allLowerBound):
                if overrideBound == True:
                    modelObject.reactions[i].lower_bound = indexLowerBound
                else:
                    if aggregateFun == 'min':
                        lb = np.min([modelObject.reactions[i].lower_bound, 
                            coef * indexLowerBound])
                    if aggregateFun == 'max':
                        lb = np.max([modelObject.reactions[i].lower_bound, 
                            coef * indexLowerBound])
                    if aggregateFun == 'mean':
                        lb = np.mean([modelObject.reactions[i].lower_bound, 
                            coef * indexLowerBound])
                    modelObject.reactions[i].lower_bound = lb 
            else:
                modelObject.reactions[i].lower_bound = indexLowerBound
            
            if (modelObject.reactions[i].upper_bound != 0 and 
                modelObject.reactions[i].upper_bound != allUpperBound):
                if overrideBound == True:
                    modelObject.reactions[i].upper_bound = indexUpperBound
                else:
                    if aggregateFun == 'min':
                        ub = np.min([modelObject.reactions[i].upper_bound, 
                            coef * indexUpperBound])
                    if aggregateFun == 'max':
                        ub = np.max([modelObject.reactions[i].upper_bound, 
                            coef * indexUpperBound])
                    if aggregateFun == 'mean':
                        ub = np.mean([modelObject.reactions[i].upper_bound, 
                            coef * indexUpperBound])
                    modelObject.reactions[i].upper_bound = ub 
            else:
                modelObject.reactions[i].upper_bound = indexUpperBound
    
    if finishScale == True:
        modelObject = geo.prepModelConstraints(modelObject, 
            lowerBound = allLowerBound, upperBound = allUpperBound)
        finalUnis = len(set([x.upper_bound for x in modelObject.reactions]))
    
    if finalUnis == origUnis and len(rxnIndices) > 0:
        print('Warning. No new bounds appeared after constrainRxnIndices()')
    
    return modelObject


def kcatAggregate(rxnDict, kcatDict, ecKey = 'EC', pathwayKey = 'Reaction pathways', 
    pathwayType = 'KEGG', logKCat = True, scaleKCat = True, scaleVal = 1000, 
    matchECOnly = False, ECPreference = True, ECAggregate = 'median', 
    matchPathwayOnly = False, choosePathway = False, pathwayChoice = 'max', 
    pathwayChoiceStat = 'median', kcatAggregate = 'median', **params):
    """ Aggregate kcats from gold standard data with the rxnDict generated
        with rxnPathwayInfo(). This function depends on the output from
        rxnPathwayInfo(), with each reaction containing fields for "EC" and
        "Pathway". Note that if scaling is requested, all kcats from the
        gold standard are scaled before GEM-matching, instead of scaling at
        the end from only kcats which were matched. Scaing is performed after
        a log-transform call. The rxnDict and kcatDict should come from the
        KinetConstrain set of functions.

        Args:
            rxnDict: Dict - Reaction dictionary from rxnPathwayInfo()
            kcatDict: Dict - KCats reference, via wrangleGoldStandard()
            logKat: Bool - Log-transform kcats from gold standard data
            scale: Bool - Scale kcats from gold standard data
            scaleVal: Bool - Scale max value
            matchECOnly: Bool - Only match on ECs, skip pathways
            ECPreference: Bool - First check for EC matches, and if none exist
                then check pathways. 
            ECAggregate: Str - Aggregation function for multiple EC values
            matchPathwayOnly: Bool - Only match pathways, skip ECs
            choosePathway: Bool - If multiple pathways are present, choose values
                from one before aggregation
            pathwayChoice: Str - If a pathway is chosen, how should it be chosen?
                'max' or 'min' for now
            pathwayChoiceStat: Str - If pathwayChoice == 'min', 
                pathwayChoiceStat == 'median' indicates the lowest median 
                kcat value among the multiple pathways present
            kcatAggregate: Str - Aggregation function for multiple kcats

        Returns:
            rxnDict, with added kcat value
    """
    if ecKey not in kcatDict:
        raise KeyError('{0} not in kcat dictionary'.format(ecKey))
    if pathwayKey not in kcatDict:
        raise KeyError('{0} not in the kcat dictionary'.format(pathwayKey))
    if pathwayType not in kcatDict[pathwayKey]:
        raise KeyError('{0} not found under the pathway ID types in the '
            'kcat dictionary'.format(pathwayType))
    
    if matchECOnly == True and matchPathwayOnly == True:
        print('Warning: you chose match EC only and pathway only, will '
            'instead set both to false to check both')
        matchECOnly, matchPathwayOnly = False, False
    
    if scaleVal != 1000:
        print('Warning, you opted for a scaling which deivates from the '
            'expected use case of [0,1000] flux bounds')
    
    ECAggregate = ECAggregate.lower()
    pathwayChoiceStat = pathwayChoiceStat.lower()
    kcatAggregate = kcatAggregate.lower()
    pathwayChoice = pathwayChoice.lower()
    if (ECAggregate not in ['min', 'mean', 'median', 'max'] or 
        kcatAggregate not in ['min', 'mean', 'median', 'max'] or
        pathwayChoiceStat not in ['min', 'mean', 'median', 'max']):
        raise ValueError('Aggregation functions must be one of ["min", "mean", '
            '"median", "max"]')
    if pathwayChoice not in ['min', 'max']: 
        raise ValueError('Pathway choice must be one of ["min", "max"]')

    kcatDict = convertKCatReference(kcatDict, logKCat = logKCat, scaleKCat = scaleKCat, 
        ecKey = ecKey, pathwayKey = pathwayKey, pathwayType = pathwayType, 
        scaleVal = scaleVal)

    for rxn in rxnDict:
        if 'EC' in rxnDict[rxn]:
            ecs = list(set(rxnDict[rxn]['EC']))
            ecKats = []
            for ec in ecs:
                if ec in kcatDict[ecKey]:
                    ecKats.append(kcatDict[ecKey][ec])
            rxnDict[rxn]['ECKats'] = ecKats
        
        if 'Pathways' in rxnDict[rxn]:
            paths = list(set(rxnDict[rxn]['Pathways']))
            pathKats = []
            for path in paths:
                if path in kcatDict[pathwayKey][pathwayType]:
                    if choosePathway == False:
                            pathKats += kcatDict[pathwayKey][pathwayType][path]
                    else:
                        pathKats.append(kcatDict[pathwayKey][pathwayType][path])
            
            if choosePathway == True and len(pathKats) > 1:
                if pathwayChoice == 'max':
                    if pathwayChoiceStat == 'max':
                        pathKats = [x for x in pathKats if np.max(x) == np.max([np.max(x) 
                            for x in pathKats])][0]
                    if pathwayChoiceStat == 'median':
                            pathKats = [x for x in pathKats if np.median(x) == np.max([np.median(x)
                                 for x in pathKats])][0]
                    if pathwayChoiceStat == 'min':
                            pathKats = [x for x in pathKats if np.min(x) == np.max([np.min(x) 
                                for x in pathKats])][0]
                    if pathwayChoiceStat == 'mean':
                            pathKats = [x for x in pathKats if np.mean(x) == np.max([np.mean(x) 
                                for x in pathKats])][0]
                
                if pathwayChoice == 'min':
                    if pathwayChoiceStat == 'max':
                        pathKats = [x for x in pathKats if np.max(x) == np.min([np.max(x) for x in pathKats])][0]
                    if pathwayChoiceStat == 'median':
                            pathKats = [x for x in pathKats if np.median(x) == np.min([np.median(x) for x in pathKats])][0]
                    if pathwayChoiceStat == 'min':
                            pathKats = [x for x in pathKats if np.min(x) == np.min([np.min(x) for x in pathKats])][0]
                    if pathwayChoiceStat == 'mean':
                            pathKats = [x for x in pathKats if np.mean(x) == np.min([np.mean(x) for x in pathKats])][0]
                
            rxnDict[rxn]['PathKats'] = pathKats
    
    if matchECOnly == True:
        for rxn in rxnDict:
            if 'ECKats' in rxnDict[rxn]:
                if len(rxnDict[rxn]['ECKats']) > 0:
                    if kcatAggregate == 'max':
                        rxnDict[rxn]['kcat'] = np.max(rxnDict[rxn]['ECKats'])
                    if kcatAggregate == 'median':
                        rxnDict[rxn]['kcat'] = np.median(rxnDict[rxn]['ECKats'])
                    if kcatAggregate == 'mean':
                        rxnDict[rxn]['kcat'] = np.mean(rxnDict[rxn]['ECKats'])
                    if kcatAggregate == 'min':
                        rxnDict[rxn]['kcat'] = np.min(rxnDict[rxn]['ECKats'])
        return rxnDict

    elif matchPathwayOnly == True:
        for rxn in rxnDict:
            if 'PathKats' in rxnDict[rxn]:
                if len(rxnDict[rxn]['PathKats']) > 0:
                    if kcatAggregate == 'max':
                        rxnDict[rxn]['kcat'] = np.max(rxnDict[rxn]['PathKats'])
                    if kcatAggregate == 'median':
                        rxnDict[rxn]['kcat'] = np.median(rxnDict[rxn]['PathKats'])
                    if kcatAggregate == 'mean':
                        rxnDict[rxn]['kcat'] = np.mean(rxnDict[rxn]['PathKats'])
                    if kcatAggregate == 'min':
                        rxnDict[rxn]['kcat'] = np.min(rxnDict[rxn]['PathKats'])
        return rxnDict

    else:
        if ECPreference == True: 
            for rxn in rxnDict:
                if 'ECKats' in rxnDict[rxn]:
                    if len(rxnDict[rxn]['ECKats']) > 0:
                        if kcatAggregate == 'max':
                            rxnDict[rxn]['kcat'] = np.max(rxnDict[rxn]['ECKats'])
                        if kcatAggregate == 'median':
                            rxnDict[rxn]['kcat'] = np.median(rxnDict[rxn]['ECKats'])
                        if kcatAggregate == 'mean':
                            rxnDict[rxn]['kcat'] = np.mean(rxnDict[rxn]['ECKats'])
                        if kcatAggregate == 'min':
                            rxnDict[rxn]['kcat'] = np.min(rxnDict[rxn]['ECKats'])
                    else:
                        if 'PathKats' in rxnDict[rxn]:
                            if len(rxnDict[rxn]['PathKats']) > 0:
                                if kcatAggregate == 'max':
                                    rxnDict[rxn]['kcat'] = np.max(rxnDict[rxn]['PathKats'])
                                if kcatAggregate == 'median':
                                    rxnDict[rxn]['kcat'] = np.median(rxnDict[rxn]['PathKats'])
                                if kcatAggregate == 'mean':
                                    rxnDict[rxn]['kcat'] = np.mean(rxnDict[rxn]['PathKats'])
                                if kcatAggregate == 'min':
                                    rxnDict[rxn]['kcat'] = np.min(rxnDict[rxn]['PathKats'])
        else:
            for rxn in rxnDict:
                if 'PathKats' in rxnDict[rxn]:
                    if len(rxnDict[rxn]['PathKats']) > 0:
                        if kcatAggregate == 'max':
                            rxnDict[rxn]['kcat'] = np.max(rxnDict[rxn]['PathKats'])
                        if kcatAggregate == 'median':
                            rxnDict[rxn]['kcat'] = np.median(rxnDict[rxn]['PathKats'])
                        if kcatAggregate == 'mean':
                            rxnDict[rxn]['kcat'] = np.mean(rxnDict[rxn]['PathKats'])
                        if kcatAggregate == 'min':
                            rxnDict[rxn]['kcat'] = np.min(rxnDict[rxn]['PathKats'])
                    else:
                        if 'ECKats' in rxnDict[rxn]:
                            if len(rxnDict[rxn]['ECKats']) > 0:
                                if kcatAggregate == 'max':
                                    rxnDict[rxn]['kcat'] = np.max(rxnDict[rxn]['ECKats'])
                                if kcatAggregate == 'median':
                                    rxnDict[rxn]['kcat'] = np.median(rxnDict[rxn]['ECKats'])
                                if kcatAggregate == 'mean':
                                    rxnDict[rxn]['kcat'] = np.mean(rxnDict[rxn]['ECKats'])
                                if kcatAggregate == 'min':
                                    rxnDict[rxn]['kcat'] = np.min(rxnDict[rxn]['ECKats'])

        return rxnDict


def convertKCatReference(kcatDict, logKCat, scaleKCat, ecKey = 'EC',
    pathwayKey = 'Reaction pathways', pathwayType = 'KEGG', scaleVal = 1000,
    minVal = 0.01):
    """ Note that minVal is set to 0.01 to avoid instabilities with 0 """

    if logKCat == True:
        allVals = dict()
        for ec in kcatDict[ecKey]:
            kat = kcatDict[ecKey][ec]
            if kat not in allVals:
                allVals[kat] = np.log(kat)
        for path in kcatDict[pathwayKey]:
            for pathID in kcatDict[pathwayKey][path]:
                for kat in kcatDict[pathwayKey][path][pathID]:
                    if kat not in allVals:
                        allVals[kat] = np.log(kat)

        for ec in kcatDict[ecKey]:
            kcatDict[ecKey][ec] = allVals[kcatDict[ecKey][ec]]
        for path in kcatDict[pathwayKey]:
            for pathID in kcatDict[pathwayKey][path]:
                kcatDict[pathwayKey][path][pathID] = [allVals[x] for x in kcatDict[pathwayKey][path][pathID]]

    if scaleKCat == True:
        allVals = set()
        for ec in kcatDict[ecKey]:
            allVals.add(kcatDict[ecKey][ec])
        for path in kcatDict[pathwayKey]:
            for pathID in kcatDict[pathwayKey][path]:
                for kat in kcatDict[pathwayKey][path][pathID]:
                    allVals.add(kat)

        valArr = np.array(list(allVals)).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range = (minVal, scaleVal))
        valScale = scaler.fit(valArr).transform(valArr)

        convertDict = dict(zip(np.squeeze(valArr), np.squeeze(valScale)))
        for ec in kcatDict[ecKey]:
            kcatDict[ecKey][ec] = convertDict[kcatDict[ecKey][ec]]
        for path in kcatDict[pathwayKey]:
            for pathID in kcatDict[pathwayKey][path]:
                kcatDict[pathwayKey][path][pathID] = [convertDict[x] for x in kcatDict[pathwayKey][path][pathID]]

    
    return kcatDict


def constrainByKCat(modelObject, kcatDict, katConstrainFun = 'mean', 
    kcatCoef = 1., prepConstraints = True, overrideBound = False, finishScale = True, 
    allLowerBound = -1000, allUpperBound = 1000, kcatKey = 'kcat', **params):
    """ Take the EC or pathway-derived kcats from kcatAggregate(), and constrain
        the metabolic model. Similar to the index constraints, aggregation can 
        occur on already-existing bounds, and a coefficient or "weight" can be
        applied too.
    
    Args:
        modelObject: GEM object
        kcatDict: Dict - Result from kcatAggregate()
        katConstrainFun: Str - aggregation function ('min', 'max', 'mean')
        KCatCoef: Float - If reaction is already constrained, weight coefficient
            of the indexLower or indexUpperBounds
        prepConstraints: Bool - Prep model constraints first
        overrideBound: Bool - Override an existing bound, and dont aggregate. As in,
            override with a KCat from kcatDict.
        finishScale: Bool - Re-scale all values within allLower and allUpperBounds
            after constraints
        allLowerBound: Float - Lower bound for prepConstraints
        allUpperBound: Float - Upper bound for prepConstraints

    Returns:
        modelObject
    """
    if (not isinstance(kcatCoef, float) and 
        not isinstance(kcatCoef, int)):
        raise ValueError('KCatCoef need to be int or float')

    if katConstrainFun not in ['min', 'max', 'mean']:
        raise ValueError('Aggregation function must be "min", "max", or "mean"')

    if prepConstraints == True:
        modelObject = geo.prepModelConstraints(modelObject, 
            lowerBound = allLowerBound, upperBound = allUpperBound)

    for i, _ in enumerate(modelObject.reactions):
        rxnName = 'v' + str(i)
        if kcatKey in kcatDict[rxnName]:
            kcatVal = kcatDict[rxnName][kcatKey]
            if (modelObject.reactions[i].lower_bound != 0 and 
                modelObject.reactions[i].lower_bound != allLowerBound):
                if overrideBound == True:
                    modelObject.reactions[i].lower_bound = -kcatVal
                else:
                    if katConstrainFun == 'min':
                        lb = np.min([modelObject.reactions[i].lower_bound, 
                            kcatCoef * -kcatVal])
                    if katConstrainFun == 'max':
                        lb = np.max([modelObject.reactions[i].lower_bound, 
                            kcatCoef * -kcatVal])
                    if katConstrainFun == 'mean':
                        lb = np.mean([modelObject.reactions[i].lower_bound, 
                            kcatCoef * -kcatVal])
                    modelObject.reactions[i].lower_bound = lb 
            else:
                modelObject.reactions[i].lower_bound = -kcatVal
            
            if (modelObject.reactions[i].upper_bound != 0 and 
                modelObject.reactions[i].upper_bound != allUpperBound):
                if overrideBound == True:
                    modelObject.reactions[i].upper_bound = kcatVal
                else:
                    if katConstrainFun == 'min':
                        ub = np.min([modelObject.reactions[i].upper_bound, 
                            kcatCoef * kcatVal])
                    if katConstrainFun == 'max':
                        ub = np.max([modelObject.reactions[i].upper_bound, 
                            kcatCoef * kcatVal])
                    if katConstrainFun == 'mean':
                        ub = np.mean([modelObject.reactions[i].upper_bound, 
                            kcatCoef * kcatVal])
                    modelObject.reactions[i].upper_bound = ub 
            else:
                modelObject.reactions[i].upper_bound = kcatVal
    
    if finishScale == True:
        modelObject = geo.prepModelConstraints(modelObject, 
            lowerBound = allLowerBound, upperBound = allUpperBound)
    
    return modelObject

