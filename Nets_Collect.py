"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import tensorflow as tf
import numpy as np
from Nets_Only import MLP_Net, TCN_Net, CNN_Net, DANN_TCN, DANN_1DCNN, DANN_MLP
from Train_Nets import MLP_Train_Loop, DANN_MLP_Train_Loop, CNN_TCN_Regression_Train_Loop, DANN_Regression_Training_Loop, CNN_TCN_Classification_Train_Loop, DANN_Classification_Training_Loop


def Get_Net(sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, sourceTrainOperation, sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels,
           targetTrainFeatures, targetTrainClasslabels, targetTrainDomainlabels, targetTrainOperation, targetCheckFeatures, targetCheckClasslabels, targetCheckDomainlabels,
           featureType, dataTask, netType, verbose, activateDocu, pathes, hyperparameter, activationDense, activationConv, paddingConv, paddingPool,
           lossFunction, metric, lossFunctionDomain, metricDomain,
           earlyStoppingYesNo, earlyStoppingEpochs, trainEpochs, INITIAL_TRAINING_DATA):
    
    model = Generate_Net(sourceTrainFeatures=sourceTrainFeatures, sourceTrainClasslabels=sourceTrainClasslabels, featureType=featureType, dataTask=dataTask, netType=netType, 
                         hyperparameter=hyperparameter, activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool,
                         lossFunction=lossFunction, metric=metric)
    model, finalTrainEpochs, informationText = Train_Net(model=model, sourceTrainFeatures=sourceTrainFeatures, sourceTrainClasslabels=sourceTrainClasslabels, sourceTrainDomainlabels=sourceTrainDomainlabels, sourceTrainOperation=sourceTrainOperation,
                                                sourceCheckFeatures=sourceCheckFeatures, sourceCheckClasslabels=sourceCheckClasslabels, sourceCheckDomainlabels=sourceCheckDomainlabels, 
                                                targetTrainFeatures=targetTrainFeatures, targetTrainClasslabels=targetTrainClasslabels, targetTrainDomainlabels=targetTrainDomainlabels, targetTrainOperation=targetTrainOperation,
                                                targetCheckFeatures=targetCheckFeatures, targetCheckClasslabels=targetCheckClasslabels, targetCheckDomainlabels=targetCheckDomainlabels,
                                                dataTask=dataTask, netType=netType, verbose=verbose, pathCheckpoint=pathes["TEMP_CHECKPOINT_MODEL_EARLY_STOP"],
                                                hyperparameter=hyperparameter, lossFunction=lossFunction, 
                                                metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                                                featureType=featureType, earlyStoppingYesNo=earlyStoppingYesNo, earlyStoppingEpochs=earlyStoppingEpochs, trainEpochs=trainEpochs, INITIAL_TRAINING_DATA=INITIAL_TRAINING_DATA)
    
    return model, finalTrainEpochs, informationText


def Generate_Net(sourceTrainFeatures, sourceTrainClasslabels, featureType, dataTask, netType,
                 hyperparameter, activationDense, activationConv, paddingConv, paddingPool, lossFunction, metric):
    
    if featureType == "TimeSeries":
        InputLayer = sourceTrainFeatures[0].shape[1]
    elif featureType == "Points" and dataTask == "Regression":
        InputLayer = sourceTrainFeatures[0].shape[1]
    elif featureType == "Points" and dataTask == "Classification":
        InputLayer = sourceTrainFeatures.shape[1]
    if dataTask == "Regression":
        OutputLayer = 1
    elif dataTask == "Classification":
        OutputLayer = len(np.unique(sourceTrainClasslabels))
    
    if netType == "MLP":
        model = MLP_Net(InputLayer=InputLayer, OutputLayer=OutputLayer, hyperparameter=hyperparameter, activationDense=activationDense, dataTask=dataTask)
    elif netType == "CNN":
        model = CNN_Net(InputLayer=InputLayer, OutputLayer=OutputLayer, hyperparameter=hyperparameter, activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool, dataTask=dataTask)
    elif netType == "TCN": 
        model = TCN_Net(InputLayer=InputLayer, OutputLayer=OutputLayer, hyperparameter=hyperparameter, activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool, dataTask=dataTask)
    elif netType == "DANN_CNN":
        model = DANN_1DCNN(InputLayer=InputLayer, OutputLayer=OutputLayer, hyperparameter=hyperparameter, activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool, dataTask=dataTask)
    elif netType == "DANN_TCN":
        model = DANN_TCN(InputLayer=InputLayer, OutputLayer=OutputLayer, hyperparameter=hyperparameter, activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool, dataTask=dataTask)
    elif netType == "DANN_MLP":
        model = DANN_MLP(InputLayer=InputLayer, OutputLayer=OutputLayer, hyperparameter=hyperparameter, activationDense=activationDense, dataTask=dataTask)
    if not "DANN" in netType: 
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameter["learnRate"]), loss=lossFunction, weighted_metrics=[metric])
    
    return model


def Train_Net(model, sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, sourceTrainOperation, sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels,
           targetTrainFeatures, targetTrainClasslabels, targetTrainDomainlabels, targetTrainOperation, targetCheckFeatures, targetCheckClasslabels, targetCheckDomainlabels,
           dataTask, netType, verbose, pathCheckpoint, hyperparameter, lossFunction, metric, lossFunctionDomain, metricDomain,
           featureType, earlyStoppingYesNo, earlyStoppingEpochs, trainEpochs, INITIAL_TRAINING_DATA):
    
    if netType == "MLP" or netType == "CNN" or netType == "TCN":
        if INITIAL_TRAINING_DATA == "Source":
            trainFeatures=sourceTrainFeatures
            trainLabels=sourceTrainClasslabels
            checkFeatures=sourceCheckFeatures
            checkLabels=sourceCheckClasslabels
            informationText = "Trainiert auf S - Getestet auf S"
        elif INITIAL_TRAINING_DATA == "Target":
            trainFeatures=targetTrainFeatures
            trainLabels=targetTrainClasslabels
            checkFeatures=targetCheckFeatures
            checkLabels=targetCheckClasslabels
            informationText = "Trainiert auf T - Getestet auf T"
        elif INITIAL_TRAINING_DATA == "All":
            trainFeatures= np.concatenate((sourceTrainFeatures, targetTrainFeatures))
            trainLabels= np.concatenate((sourceTrainClasslabels, targetTrainClasslabels))
            checkFeatures=targetCheckFeatures
            checkLabels=targetCheckClasslabels
            informationText = "Trainiert auf S + T - Getestet auf T"
        if netType == "MLP":
            model, ownHistory, finalTrainEpochs = MLP_Train_Loop(model=model, trainFeatures=trainFeatures, trainLabels=trainLabels,
                                                            checkFeatures=checkFeatures, checkLabels=checkLabels, batchSize=hyperparameter["batchSize"], trainEpochs=trainEpochs,
                                                            earlyStoppingYesNo=earlyStoppingYesNo, earlyStoppingEpochs=earlyStoppingEpochs, dataTask=dataTask, informationText=informationText, 
                                                            pathCheckpoint=pathCheckpoint, verbose=verbose)
        elif (netType == "CNN" or netType == "TCN") and dataTask == "Regression":
            model, ownHistory, finalTrainEpochs = CNN_TCN_Regression_Train_Loop(model=model, trainFeatures=trainFeatures, trainLabels=trainLabels,
                                                            checkFeatures=checkFeatures, checkLabels=checkLabels, trainEpochs=trainEpochs, earlyStoppingYesNo=earlyStoppingYesNo,
                                                            earlyStoppingEpochs=earlyStoppingEpochs, featureType=featureType, netType=netType, metric=metric, informationText=informationText, pathCheckpoint=pathCheckpoint, verbose=verbose)
        elif (netType == "CNN" or netType == "TCN") and dataTask == "Classification":
            model, ownHistory, finalTrainEpochs = CNN_TCN_Classification_Train_Loop(model=model, trainFeatures=trainFeatures, trainLabels=trainLabels,
                                                            checkFeatures=checkFeatures, checkLabels=checkLabels, trainEpochs=trainEpochs, earlyStoppingYesNo=earlyStoppingYesNo,
                                                            earlyStoppingEpochs=earlyStoppingEpochs, featureType=featureType, netType=netType, metric=metric, informationText=informationText, pathCheckpoint=pathCheckpoint, verbose=verbose)
    elif "DANN" in netType:
        informationText = "Trainiert auf S+T - Getestet auf T"
        if "MLP" in netType:
            (model, ownHistory, finalTrainEpochs) = DANN_MLP_Train_Loop(model=model, sourceTrainFeatures=sourceTrainFeatures, sourceTrainClasslabels=sourceTrainClasslabels, sourceTrainDomainlabels=sourceTrainDomainlabels, sourceTrainOperation=sourceTrainOperation,
                                                                    targetTrainFeatures=targetTrainFeatures, targetTrainDomainlabels=targetTrainDomainlabels, targetTrainOperation=targetTrainOperation,
                                                                    sourceCheckFeatures=sourceCheckFeatures, sourceCheckClasslabels=sourceCheckClasslabels, sourceCheckDomainlabels=sourceCheckDomainlabels,
                                                                    targetCheckFeatures=targetCheckFeatures, targetCheckClasslabels=targetCheckClasslabels, targetCheckDomainlabels=targetCheckDomainlabels,
                                                                    batchSize=hyperparameter["batchSize"], trainEpochs=trainEpochs, lambdaValue=hyperparameter["lambdaValue"], learnRate=hyperparameter["learnRate"], earlyStoppingYesNo=earlyStoppingYesNo, 
                                                                    earlyStoppingEpochs=earlyStoppingEpochs, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                                                                    dataTask=dataTask, informationText=informationText, pathCheckpoint=pathCheckpoint, verbose=verbose)
        else:
            if dataTask == "Regression":
                (model, ownHistory, finalTrainEpochs) = DANN_Regression_Training_Loop(model=model, sourceTrainFeatures=sourceTrainFeatures, sourceTrainClasslabels=sourceTrainClasslabels, sourceTrainDomainlabels=sourceTrainDomainlabels,
                                                                        targetTrainFeatures=targetTrainFeatures, targetTrainDomainlabels=targetTrainDomainlabels,
                                                                        sourceCheckFeatures=sourceCheckFeatures, sourceCheckClasslabels=sourceCheckClasslabels, sourceCheckDomainlabels=sourceCheckDomainlabels,
                                                                        targetCheckFeatures=targetCheckFeatures, targetCheckClasslabels=targetCheckClasslabels, targetCheckDomainlabels=targetCheckDomainlabels,
                                                                        trainEpochs=trainEpochs, lambdaValue=hyperparameter["lambdaValue"], learnRate=hyperparameter["learnRate"], earlyStoppingYesNo=earlyStoppingYesNo, 
                                                                        earlyStoppingEpochs=earlyStoppingEpochs, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                                                                        informationText=informationText, pathCheckpoint=pathCheckpoint, verbose=verbose)
            elif dataTask == "Classification":
                (model, ownHistory, finalTrainEpochs) = DANN_Classification_Training_Loop(model=model, sourceTrainFeatures=sourceTrainFeatures, sourceTrainClasslabels=sourceTrainClasslabels, sourceTrainDomainlabels=sourceTrainDomainlabels, sourceTrainOperation=sourceTrainOperation,
                                                                      targetTrainFeatures=targetTrainFeatures, targetTrainDomainlabels=targetTrainDomainlabels, targetTrainOperation=targetTrainOperation,
                                                                      sourceCheckFeatures=sourceCheckFeatures, sourceCheckClasslabels=sourceCheckClasslabels, sourceCheckDomainlabels=sourceCheckDomainlabels,
                                                                      targetCheckFeatures=targetCheckFeatures, targetCheckClasslabels=targetCheckClasslabels, targetCheckDomainlabels=targetCheckDomainlabels,
                                                                      trainEpochs=trainEpochs, lambdaValue=hyperparameter["lambdaValue"], learnRate=hyperparameter["learnRate"], earlyStoppingYesNo=earlyStoppingYesNo, 
                                                                      earlyStoppingEpochs=earlyStoppingEpochs, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                                                                      informationText=informationText, pathCheckpoint=pathCheckpoint, verbose=verbose)
    informationText = informationText.split(" - ")[0]
    
    return model, finalTrainEpochs, informationText



