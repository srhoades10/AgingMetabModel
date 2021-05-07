""" Unit-testing for DataNormalization. If this script runs long, then make
    a secondary script to pull the GSE files only once, rather than multiple times
    across the functions. """

import unittest, sys, os, re, warnings
import pandas as pd
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_array_equal
import numpy as np
from scipy import stats
from sklearn import preprocessing, decomposition

sys.path.append('CobraCycle/src/')
import setup_GEOWrangle as filePull
import setup_DataNormalization as dataProcess
geoDir = 'CobraCycle/GEOData'

class TestDataProcess(unittest.TestCase):

    def test_processData(self):
        """ Check: MxN expression data table, where IDs are in the first column. """
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
            'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        data, _, _ = filePull.experimentFilesReader(geo)
        data = data[data.columns[1:]]
        with self.assertRaises(Exception) as msg: 
           dataProcess.processData(data)
        the_except = msg.exception
        self.assertEqual(str(the_except), 
            'Transcript IDs not in first column: check data format')

    def test_adjMinValue(self):
        """ Check: minimum value of output is in fact g.t.e. minValue. """
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        data, _, _ = filePull.experimentFilesReader(geo)
        dataOnly = data[data.columns[1:]]
        adjustValPos = 1.0
        dataOnlyAdj = dataProcess.adjMinValue(dataOnly, adjustValPos)
        self.assertGreaterEqual(np.nanmin(dataOnlyAdj.min()), adjustValPos)
        
        adjustValNeg = -1.0
        with self.assertRaises(ValueError) as msg: 
           dataProcess.adjMinValue(dataOnly, adjustValNeg)
        the_except = msg.exception
        self.assertEqual(str(the_except), 
            'Minimum value must be >= 0')

    def test_rmMissingness(self):
        """ Check: The output DataFrame shape changes when a proper cutoff is applied. """
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        dataRm1 = dataProcess.rmMissingness(data, 0.0, 0.0)
        self.assertEqual(dataRm1.shape, (0, 0))

        dataRm2 = dataProcess.rmMissingness(data, 1.0, 1.0)
        assert_frame_equal(dataRm2, data)

        data[data.columns[1]] = np.nan
        data.iloc[1] = np.nan
        dataRm3 = dataProcess.rmMissingness(data, 0.5, 0.5)
        self.assertEqual(dataRm3.shape[0], data.shape[0] - 1)
        self.assertEqual(dataRm3.shape[1], data.shape[1] - 1)

    def test_logTransform(self):
        """ Check: Data is transformed if values > 1000 exist and it doesn't result
            in skew < 0.25. Data is NOT transformed if the skew > 0.25 and is the
            same as the original data. """
        data1, _, _ = filePull.experimentFilesReader('GSE71117')

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            with self.assertRaises(Exception) as msg: 
               dataProcess.logTransform(data1, 1.0, 0.25)
            the_except = msg.exception
            self.assertEqual(str(the_except), 
                'Log-transform appears to have dropped skewness below 0.25, '
                        'original data will be kept as-is')
        dataTransform = dataProcess.logTransform(data1, 1.0, 0.25, True)
        dataTransformOnly = dataTransform[dataTransform.columns[1:]]
        
        data1Only = data1[data1.columns[1:]]
        dataCompare = dataProcess.adjMinValue(data1Only, 1.0)
        assert_frame_equal(dataTransformOnly, dataCompare)

        data2, _, _ = filePull.experimentFilesReader('GSE64785')
        data2Log = dataProcess.logTransform(data2, 1.0, 0.25, True)
        data2Only = data2Log[data2Log.columns[1:]]
        self.assertLess(np.nanmax(np.ravel(data2Only)), 1000)

    def test_imputeMissing(self):
        """ Check: Output DataFrame is of the same dimension. No nans exist. """
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _  = filePull.experimentFilesReader(geo)
        data2 = dataProcess.imputeMissing(data, 'median')
        self.assertEqual(data.shape, data2.shape)
        dataOnly = data2[data2.columns[1:]]
        self.assertFalse(any(np.isnan(np.ravel(dataOnly))))
    
    def test_calc2DHotellings(self):
        """ Check: Cholesky decomposition can be made. x > y. dummy positive control
             dataset gives expected t1 and t2. """
        _, posToyProjections = dataProcess.createPosToyPCAData(True)
        _, negToyProjections = dataProcess.createNegToyPCAData()
        radPosX, radPosY = dataProcess.calc2DHotellings(posToyProjections, 99.7)
        self.assertAlmostEqual(6.6, round(radPosX, 2))
        self.assertAlmostEqual(2.4, round(radPosY, 1))

        with self.assertRaises(ArithmeticError) as msg: 
            dataProcess.calc2DHotellings(negToyProjections, 99.7)
        the_except = msg.exception
        self.assertEqual(str(the_except), 
            'Cholesky failed, check PCA for successful SVD and '
            'that there arent columns with 0 variance')
        
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        #take some code from rmPCAOutliers
        dataOnly = data[data.columns[1:]].T #NxM
        dataOnly.reset_index(inplace = True, drop = True)
        dataProjections = dataProcess.fitPCA(dataOnly)
        radiusX, radiusY = dataProcess.calc2DHotellings(dataProjections, 99.7)
        self.assertGreaterEqual(radiusX, radiusY)

    def test_flag2DOutliers(self):
        """ Check: In a toy dataset, outliers are successfully removed with PCA """
        _, posToyProjections = dataProcess.createPosToyPCAData(True)
        _, negToyProjections = dataProcess.createPosToyPCAData(False)
        radPosX, radPosY = dataProcess.calc2DHotellings(posToyProjections, 99.7)
        radNegX, radNegY = dataProcess.calc2DHotellings(negToyProjections, 99.7)
        posOutliers = dataProcess.flag2DOutliers(posToyProjections, radPosX, radPosY)
        negOutliers = dataProcess.flag2DOutliers(negToyProjections, radNegX, radNegY)
        self.assertEqual(posOutliers, [100, 101, 102, 103, 104])
        self.assertEqual(negOutliers, [])

    def test_rmPCAOutliers(self):
        """ Check: Outliers are flagged and removed on a real data set. Improper 
            hotelling cutoff value is caught with a warning. """
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)

        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            with self.assertRaises(Exception) as msg: 
               dataProcess.rmPCAOutliers(data, 0.95)
            the_except = msg.exception
            self.assertEqual(str(the_except), 
                'Warning: percentileCutoff is between 0 and 1, did you mean '
                'to enter such a low percentage? You can enter any number between '
                '0 and 100, with 95 representing 2 SDs')
        trimDataCtl = dataProcess.rmPCAOutliers(data, 99.999)
        self.assertEqual(data.shape, trimDataCtl.shape)
        assert_frame_equal(data, trimDataCtl)
        trimDataExp = dataProcess.rmPCAOutliers(data, 68)
        self.assertEqual(data.shape[0], trimDataExp.shape[0])
        self.assertGreater(data.shape[1], trimDataExp.shape[1])

    def test_filterCV(self):
        """ Check: absoluteCVCutoff removes probes. Probes with CVs between 0 and
            meanCVThreshold and below the dataset mean value have zero variance, and
            those above the dataset mean value do not (meaning they weren't touched) """
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        data2 = dataProcess.filterCV(data, 100)
        self.assertGreater(data.shape[0], data2.shape[0])
        data3 = dataProcess.filterCV(data, 10000)
        self.assertEqual(data.shape[0], data3.shape[0])
    
    def test_calcCVs(self):
        """ Check: Row-wise CV can be properly calculated """
        toyData = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        testMeans = np.mean(toyData[:, 1:], axis = 1)
        testCVs = np.std(toyData[:, 1:], axis = 1) / testMeans * 100
        CVs, probeMeans = dataProcess.calcCVs(pd.DataFrame(toyData))
        assert_array_equal(testMeans, probeMeans)
        assert_array_equal(testCVs, CVs)
    
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
            'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        dataTransform = dataProcess.logTransform(data, 1.0, 0.25)
        dataOnly = dataTransform[dataTransform.columns[1:]]
        CVs, probeMeans = dataProcess.calcCVs(dataTransform)
        self.assertEqual(len(CVs), len(dataOnly))
        self.assertEqual(len(probeMeans), len(dataOnly))
        self.assertGreaterEqual(min(CVs), 0)
        self.assertGreaterEqual(min(probeMeans), 0)

    def test_setMeanProbeVals(self):
        """ Check: Probes with low expression and high CVs have their values set
            to their means, meaning an array of S.D.s of all rows in the original
            data must be greater than 0, and an array or S.D.s of all rows after
            must contain zeros """
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
            'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        dataTransform = dataProcess.logTransform(data, 1.0, 0.25, True)
        dataOnly = dataTransform[dataTransform.columns[1:]]
        origSDs = np.nanstd(dataOnly, axis = 1)
        self.assertEqual(0, len(origSDs[origSDs == 0]))

        CVs, probeMeans = dataProcess.calcCVs(dataOnly)
        averagedData = dataProcess.setMeanProbeVals(dataTransform, 20, probeMeans, CVs)
        averagedDataOnly = averagedData[averagedData.columns[1:]]
        averagedSDs = np.nanstd(averagedDataOnly, axis = 1)
        self.assertLess(0, len(averagedSDs[averagedSDs == 0]))

    def test_calcMAD(self):
        """ Check: the MAD calcuations are accurate, including the lambda function
            inside calcMAD(). """
        testArray = np.array([1, 1, 2, 2, 4, 6, 9])
        subCalc = lambda x: np.median(abs(x - np.median(x)))
        testResult = subCalc(testArray)
        self.assertEqual(1.0, testResult)


    def test_adjustMedianScaling(self):
        """ Check: median-based scaling properly calculates on control array.
            Scaling results in new values. Value ordering is consistent pre and post
            scaling (checkArgsort = True). Inverse correlation of skew with coef."""
        testArray = np.array([1, 1, 2, 2, 4, 6, 9])
        testResult = np.array([0.875, 0.875, 2, 2, 5, 9, 16.875])
        assert_array_equal(testResult, dataProcess.adjustMedianScaling(testArray))
        
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        data = dataProcess.rmMissingness(data, 0.25, 0.25)
        data = dataProcess.imputeMissing(data, 'median')
        data = dataProcess.logTransform(data, 1.0, 0.25, True)
        data = dataProcess.rmPCAOutliers(data, 99.7)
        data = dataProcess.filterCV(data, 200)
        dataOnly = data[data.columns[1:]]
        
        randomProbes = np.random.randint(0, len(dataOnly), 100)
        for probe in randomProbes:
            testProbe = dataOnly.iloc[probe]
            baseAdjust = dataProcess.adjustMedianScaling(testProbe)
            self.assertNotEqual(min(testProbe), min(baseAdjust))
            self.assertNotEqual(max(testProbe), max(baseAdjust))
            weakAdjust = dataProcess.adjustMedianScaling(testProbe, coef = 8, 
                checkArgsort = True)
            self.assertLess(stats.skew(weakAdjust), stats.skew(baseAdjust))
            strongAdjust = dataProcess.adjustMedianScaling(testProbe, coef = 1, 
                checkArgsort = True)
            self.assertGreater(stats.skew(strongAdjust), stats.skew(baseAdjust))


    def test_scaleExpression(self):
        """ Check: Output data dimension is the same, regardless of axis choice.
            Changing the coefficient actually reuslts in different data. Proper
            coefficient values are passed in, and improper are rejected. """         
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        data = dataProcess.rmMissingness(data, 0.25, 0.25)
        data = dataProcess.imputeMissing(data, 'median')
        data = dataProcess.logTransform(data, 1.0, 0.25, True)
        data = dataProcess.rmPCAOutliers(data, 99.7)
        data = dataProcess.filterCV(data, 200)

        with self.assertRaises(ValueError) as msg: 
            dataProcess.scaleExpression(data, axis = 1, coef = -1)
        the_except = msg.exception
        self.assertEqual(str(the_except), 
            'CV cutoff must be a numeric positive real')

        weakAdjust = dataProcess.scaleExpression(data, axis = 1, coef = 5.5, 
            suppressWarning = True)
        self.assertEqual(data.shape, weakAdjust.shape)
        strongAdjust = dataProcess.scaleExpression(data, axis = 1, coef = 2.5, 
            suppressWarning = True)
        self.assertEqual(data.shape, strongAdjust.shape)

        weakAdjustData = weakAdjust[weakAdjust.columns[1:]]
        strongAdjustData = strongAdjust[strongAdjust.columns[1:]]
        self.assertLess(np.max(np.ravel(weakAdjustData)), 
            np.max(np.ravel(strongAdjustData)))

    def test_exprToPercentile(self):
        """ Check: Not currently in use, but ensure values range from 0-1000. """         
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        data = dataProcess.rmMissingness(data, 0.25, 0.25)
        data = dataProcess.imputeMissing(data, 'median')
        data = dataProcess.logTransform(data, 1.0, 0.25, True)
        data = dataProcess.exprToPercentile(data)
        dataOnly = data[data.columns[1:]]
        allVals = np.ravel(dataOnly)
        self.assertEqual(0., round(min(allVals), 1))
        self.assertEqual(1000., round(max(allVals), 1))

    def test_scaleCBM(self):
        """ Check: minimum value post-scaling is roughly 0, maximum value
            post-scaling is roughly 1000. A random sample of values pre and post
            scaling exhibit the same argsort() vector, meaning reraveling the data
            in the original data shape maintained the right axes """
        geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
            'GSE71117', 'GSE64785']        
        geo = np.random.choice(geoList)
        
        data, _, _ = filePull.experimentFilesReader(geo)
        data = dataProcess.rmMissingness(data, 0.25, 0.25)
        data = dataProcess.imputeMissing(data, 'median')
        data = dataProcess.logTransform(data, 1.0, 0.25, True)
        data = dataProcess.rmPCAOutliers(data, hotellings = 99.7)
        data = dataProcess.filterCV(data, 200)
        data = dataProcess.scaleExpression(data, axis = 1, coef = 4)
        dataOnly = data[data.columns[1:]]
        dataScale = dataProcess.scaleCBM(data, 'ID_REF', 0, 1000)
        dataScaleOnly = dataScale[dataScale.columns[1:]]
        allScaleVals = np.ravel(dataScaleOnly)
        self.assertEqual(0., round(min(allScaleVals), 1))
        self.assertEqual(1000., round(max(allScaleVals), 1))

        randomProbes = np.random.randint(0, len(dataOnly), 100)
        for probe in randomProbes:
            testProbePre = dataOnly.iloc[probe]
            testProbePost = dataScaleOnly.iloc[probe]
            assert_array_equal(np.argsort(testProbePre), np.argsort(testProbePost))


if __name__ == '__main__':

    geoList = ['GSE19109', 'GSE37307', 'GSE54536', 'GSE59941', 'GSE9785', 'GSE8708',
        'GSE71117', 'GSE64785']
    for geo in geoList:
        if not any([re.search(geo, x) for x in os.listdir(geoDir)]):
            readIn, _, _ = filePull.experimentFilesReader(geo)

    unittest.main()
