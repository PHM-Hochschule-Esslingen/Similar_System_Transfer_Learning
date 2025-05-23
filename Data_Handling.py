"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random


def Normalize_Data(signal, whatToDo, norm=[]):
    
    if whatToDo == "standPara" or whatToDo == "norm01Para" or whatToDo == "norm-11Para":
        norm = {}
        if whatToDo == "standPara":
            norm["mean"] = np.mean(signal, axis=0)
            norm["std"] = np.sqrt(np.var(signal, axis=0))
        elif whatToDo == "norm01Para":
            norm["maxValues"] = np.max(signal, axis=0)
            norm["minValues"] = np.min(signal, axis=0)
        elif whatToDo == "norm-11Para":
            norm["maxValues"] = np.max(signal, axis=0)
            norm["minValues"] = np.min(signal, axis=0)
        return norm
    
    if whatToDo == "stand" or whatToDo == "norm01" or whatToDo == "norm-11":
        if whatToDo == "stand":
            zeroDev = (norm["std"] == 0)
            signal[:,~zeroDev] = (signal[:,~zeroDev] - norm["mean"][~zeroDev]) / norm["std"][~zeroDev]
            signal[:,zeroDev] = signal[:,zeroDev] * 0
        elif whatToDo == "norm01":
            zeroDev = norm["maxValues"] == norm["minValues"]
            signal[:,~zeroDev] = (signal[:,~zeroDev] - norm["minValues"][:,~zeroDev]) / ( norm["maxValues"][:,~zeroDev] - norm["minValues"][:,~zeroDev] )
            signal[:,zeroDev] = signal[:,zeroDev] * 0
        elif whatToDo == "norm-11":
            zeroDev = norm["maxValues"] == norm["minValues"]
            signal[:,~zeroDev] = ( signal[:,~zeroDev] - (norm["minValues"][:,~zeroDev] + norm["maxValues"][:,~zeroDev])/2 ) / ( (norm["maxValues"][:,~zeroDev] - norm["minValues"][:,~zeroDev] )/2 );
            signal[:,zeroDev] = signal[:,zeroDev] * 0
        return signal
    
    elif whatToDo == "Rstand" or whatToDo == "Rnorm01" or whatToDo == "Rnorm-11":
        if whatToDo == "Rstand":
            signal = signal * norm["std"] + norm["mean"]
        elif whatToDo == "Rnorm01":
            signal = signal * ( norm["maxValues"] - norm["minValues"] ) + norm["minValues"]
        elif whatToDo == "Rnorm-11":
            signal = signal * ( (norm["maxValues"] - norm["minValues"])/2 ) + (norm["minValues"] + norm["maxValues"])/2   
        return signal


def Devide_Data_And_Substitute_Labels(features, labels, domains, optCond, nameOfTargetDataset, sourceTestSize, targetTestSize, sourceValSize, targetValSize, 
                                      task, featureType, fairSplitFaults, fairSplitOperationCond, howManyexcludedOptCondTargetTrainVal):
    
    if task == "Classification":
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
    if task == "Regression":
        stratifyValues = domains
    elif task == "Classification":
        stratifyValues = domains
        if fairSplitFaults == True:
            stratifyValues = stratifyValues + "_fault" + labels.astype(str)
        if fairSplitOperationCond == True:
            stratifyValues = stratifyValues + "_oCond" + optCond
    
    (sourceTrainIdx, sourceTestIdx, sourceValIdx,
    targetTrainIdx, targetTestIdx, targetValIdx) = Get_Source_Target_Train_Test_Val_Idx(domains=domains, nameOfTargetDataset=nameOfTargetDataset,
                                                      sourceTestSize=sourceTestSize, targetTestSize=targetTestSize,
                                                      sourceValSize=sourceValSize, targetValSize=targetValSize, stratifyValues=stratifyValues)

    if task == "Classification" and (howManyexcludedOptCondTargetTrainVal is not None):
        uniqueOptCond = list(set(optCond))
        selOptCond = random.sample(uniqueOptCond, howManyexcludedOptCondTargetTrainVal)
        excludeIndexes = [i for i, element in enumerate(optCond) if element in selOptCond]
        targetTrainIdx = [item for item in targetTrainIdx if item not in excludeIndexes]
        targetValIdx = [item for item in targetValIdx if item not in excludeIndexes]
    if featureType == "Points" and task == "Classification":
        sourceTrainFeatures = features[sourceTrainIdx, :]
        sourceTestFeatures = features[sourceTestIdx, :]
        sourceValFeatures = features[sourceValIdx, :]
        targetTrainFeatures = features[targetTrainIdx, :]
        targetTestFeatures = features[targetTestIdx, :]
        targetValFeatures = features[targetValIdx, :]
    else:
        sourceTrainFeatures = features[sourceTrainIdx]
        sourceTestFeatures = features[sourceTestIdx]
        sourceValFeatures = features[sourceValIdx]
        targetTrainFeatures = features[targetTrainIdx]
        targetTestFeatures = features[targetTestIdx]
        targetValFeatures = features[targetValIdx]
    
    sourceTrainClasslabels = labels[sourceTrainIdx]
    sourceTestClasslabels = labels[sourceTestIdx]
    sourceValClasslabels = labels[sourceValIdx]
    targetTrainClasslabels = labels[targetTrainIdx]
    targetTestClasslabels = labels[targetTestIdx] 
    targetValClasslabels = labels[targetValIdx] 
    sourceTrainDomainlabels = np.ones(sourceTrainClasslabels.shape, dtype = int)
    sourceTestDomainlabels = np.ones(sourceTestClasslabels.shape, dtype = int)
    sourceValDomainlabels = np.ones(sourceValClasslabels.shape, dtype = int)
    targetTrainDomainlabels = np.zeros(targetTrainClasslabels.shape, dtype = int)
    targetTestDomainlabels = np.zeros(targetTestClasslabels.shape, dtype = int)
    targetValDomainlabels = np.zeros(targetValClasslabels.shape, dtype = int)
    
    return sourceTrainFeatures, sourceTestFeatures, sourceValFeatures, sourceTrainClasslabels, sourceTestClasslabels, sourceValClasslabels, sourceTrainDomainlabels, sourceTestDomainlabels, sourceValDomainlabels,\
    targetTrainFeatures, targetTestFeatures, targetValFeatures, targetTrainClasslabels, targetTestClasslabels, targetValClasslabels, targetTrainDomainlabels, targetTestDomainlabels, targetValDomainlabels,\
    sourceTrainIdx, targetTrainIdx, sourceTestIdx, targetTestIdx, sourceValIdx, targetValIdx, le if 'le' in locals() else None


def Get_Source_Target_Train_Test_Val_Idx(domains, nameOfTargetDataset, sourceTestSize, targetTestSize, sourceValSize, targetValSize, stratifyValues=None):
   
    sourceIdx = [i for i, x in enumerate(domains != nameOfTargetDataset) if x]
    targetIdx = [i for i, x in enumerate(domains == nameOfTargetDataset) if x]
    sourceStratifyValues = stratifyValues[sourceIdx]
    targetStratifyValues = stratifyValues[targetIdx]
    n_total = len(sourceIdx)
    n_val = int(sourceValSize * n_total)
    n_test = int(sourceTestSize * n_total)
    n_train = n_total - n_val - n_test
    sourceTrainIdx, sourceTestIdx = train_test_split(sourceIdx, test_size=n_test, stratify=sourceStratifyValues)
    sourceTrainStratifyValues = stratifyValues[sourceTrainIdx]
    sourceTrainIdx, sourceValIdx = train_test_split(sourceTrainIdx, train_size=n_train, stratify=sourceTrainStratifyValues)
    n_total = len(targetIdx)
    n_val = int(targetValSize * n_total)
    n_test = int(targetTestSize * n_total)
    n_train = n_total - n_val - n_test
    targetTrainIdx, targetTestIdx = train_test_split(targetIdx, test_size=n_test, stratify=targetStratifyValues)
    targetTrainStratifyValues = stratifyValues[targetTrainIdx]
    targetTrainIdx, targetValIdx = train_test_split(targetTrainIdx, train_size=n_train, stratify=targetTrainStratifyValues)
    
    return sourceTrainIdx, sourceTestIdx, sourceValIdx, targetTrainIdx, targetTestIdx, targetValIdx


def Transform_Timeseries_To_Points_For_MLP(features, labels=None, domainLabels=None):
    
    featureArray = np.concatenate((features))
    if labels is not None:
        RULArray = np.concatenate((labels))
    if domainLabels is not None:
        domainLabelsExpanded = np.empty((len(domainLabels),), dtype=object)
        for currSeries in range(len(domainLabels)):
            currSeriesLen = len(features[currSeries])
            domainLabelsExpanded[currSeries] = np.ones(currSeriesLen) * domainLabels[currSeries]
        DomainArray = np.concatenate((domainLabelsExpanded))
    shuffle = np.random.permutation(featureArray.shape[0])
    featureArray = featureArray[shuffle]
    if labels is not None:
        RULArray = RULArray[shuffle]
    if domainLabels is not None:
        DomainArray = DomainArray[shuffle]
    
    if domainLabels is not None and labels is not None: 
        return featureArray, RULArray, DomainArray
    elif labels is None:
        return featureArray, DomainArray
    else:
        return featureArray, RULArray
