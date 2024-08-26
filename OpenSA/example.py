"""
    -*- coding: utf-8 -*-
    @Time   :2022/04/12 17:10
    @Author : Pengyou FU
    @blogs  : https://blog.csdn.net/Echo_Code?spm=1000.2115.3001.5343
    @github : https://github.com/FuSiry/OpenSA
    @WeChat : Fu_siry
    @License：Apache-2.0 license

"""


import numpy as np
from DataLoad.DataLoad import SetSplit, LoadNirtest
from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from Plot.SpectrumPlot import plotspc
# from Plot.SpectrumPlot import ClusterPlot
from Simcalculation.SimCa import Simcalculation
from Clustering.Cluster import Cluster
from Regression.Rgs import QuantitativeAnalysis
from Classification.Cls import QualitativeAnalysis

# Spectral clustering analysis
def SpectralClusterAnalysis(data, label, ProcessMethods, FslecetedMethods, ClusterMethods):
    """
     :param data: shape (n_samples, n_features), spectral data
     :param label: shape (n_samples, ), labels corresponding to the spectral data (physical and chemical properties)
     :param ProcessMethods: string, string, preprocessing method; refer to the preprocessing module for details
     :param FslecetedMethods: string,  string, spectral wavelength selection method; options include UVE, SPA, Lars, Cars, Pca
     :param ClusterMethods : string, clustering method; options include K-means clustering, FCM clustering
     :return: Clusterlabels: the returned membership matrix

     """
    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, _ = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    Clusterlabels = Cluster(ClusterMethods, FeatrueData)
    #ClusterPlot(data, Clusterlabels)
    return Clusterlabels

# Spectral quantitative analysis
def SpectralQuantitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model):

    """
    :param data: shape (n_samples, n_features), spectral data
    :param label: shape (n_samples, ), labels corresponding to the spectral data (physical and chemical properties)
    :param ProcessMethods: string, preprocessing method; refer to the preprocessing module for details
    :param FslecetedMethods: string, spectral wavelength selection method; options include UVE, SPA, Lars, Cars, Pca
    :param SetSplitMethods : string, dataset splitting method; options include random split, KS split, SPXY split
    :param model : string, quantitative analysis model; includes ANN, PLS, SVR, ELM, CNN, SAE, etc. This list will be updated over time
    :return: Rmse: float, RMSE regression error evaluation metric
             R2: float, regression fit
             Mae: float, MAE regression error evaluation metric
    """
    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    Rmse, R2, Mae = QuantitativeAnalysis(model, X_train, X_test, y_train, y_test )
    return Rmse, R2, Mae

# Spectral Qualitative Analysis
def SpectralQualitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model):

    """
    :param data: shape (n_samples, n_features), spectral data
    :param label: shape (n_samples, ), labels corresponding to spectral data (physical and chemical properties)
    :param ProcessMethods: string, preprocessing methods, refer to the preprocessing module for details
    :param FslecetedMethods: string, methods for spectral wavelength selection, options include UVE, SPA, Lars, Cars, Pca
    :param SetSplitMethods: string, methods for dataset splitting, options include random splitting, KS splitting, SPXY splitting
    :param model: string, qualitative analysis models, including ANN, PLS_DA, SVM, RF, CNN, SAE, etc., with more to be added
    :return: acc: float, classification accuracy
    """

    ProcesedData = Preprocessing(ProcessMethods, data)
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label)
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2, randomseed=123)
    acc = QualitativeAnalysis(model, X_train, X_test, y_train, y_test )

    return acc









if __name__ == '__main__':

    # ## Load raw data and visualize
    # data1, label1 = LoadNirtest('Cls')
    # #plotspc(data1, "raw spectrum")
    # # Spectral Qualitative Analysis demonstration
    # # Example 1: Preprocessing algorithm: MSC, Wavelength selection algorithm: None, Full wavelength modeling, Dataset splitting: Random splitting, Qualitative analysis model: RF
    # acc = SpectralQualitativeAnalysis(data1, label1, "MSC", "Lars", "random", "PLS_DA")
    # print("The acc:{} of result!".format(acc))


   ## Load raw data and visualize
    data2, label2 = LoadNirtest('Rgs')
    plotspc(data2, "raw specturm")
    # Spectral Quantitative Analysis demonstration
    # Example 1: Preprocessing algorithm: MSC, Wavelength selection algorithm: Uve, Dataset splitting: KS, Qualitative analysis model: SVR
    RMSE, R2, MAE = SpectralQuantitativeAnalysis(data2, label2, "None", "None", "random", "CNN")
    print("The RMSE:{} R2:{}, MAE:{} of result!".format(RMSE, R2, MAE))



      # ## Spectral preprocessing and visualization
    # method = "SNV"
    # Preprocessingdata = Preprocessing(method, data)
    # plotspc(Preprocessingdata, method)
    # ## Wavelength feature selection and visualization
    # method = 'Uve'
    # SpectruSelected, y = SpctrumFeatureSelcet(method, data, label)
    # print("Full spectrum data dimensions")
    # print(len(data[0,:]))
    # print("Data dimensions after {} wavelength selection".format(method))
    # print(len(SpectruSelected[0, :]))
    # # # Split the dataset
    # X_train, X_test, y_train, y_test = SetSplit('spxy', SpectruSelected, y, 0.2, 123)




