"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import tensorflow as tf
from Train_Nets import MLP_Train_Loop, CNN_TCN_Regression_Train_Loop, CNN_TCN_Classification_Train_Loop


def Retrain_Transfer_Learning(model, dataTask, lastLayersRetrainPercentage, fineTuningYesNo, retrainYesNo, verbose, targetTrainFeatures, targetTrainClasslabels, targetCheckFeatures, targetCheckClasslabels,
                              trainEpochsRetrain, batchSize, netType, learnRate, fineTuningLearnRate, trainEpochsFinetuning, lossFunction, metric, pathCheckpoint, informationText, 
                              featureType, earlyStoppingYesNo, earlyStoppingEpochs):
    
    for layer in model.layers:
        layer.trainable = True
        
    if retrainYesNo == True:
        informationText = informationText + ", Retrain auf T - Getestet auf T"
        numLastLayersRetrained = round(lastLayersRetrainPercentage * len(model.layers))
        for layer in model.layers[:-numLastLayersRetrained]:
            layer.trainable = False
        for i, layer in enumerate(model.layers):
            layer._name = 'Layer' + str(i)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learnRate), loss=lossFunction, metrics=[metric])
        if netType == "MLP":
            model, ownHistoryRetraining, finalTrainEpochsRetrain = MLP_Train_Loop(model=model, trainFeatures=targetTrainFeatures, trainLabels=targetTrainClasslabels,
                                                                                  checkFeatures=targetCheckFeatures, checkLabels=targetCheckClasslabels,
                                                                                  batchSize=batchSize, trainEpochs=trainEpochsRetrain, earlyStoppingYesNo=earlyStoppingYesNo,
                                                                                  earlyStoppingEpochs=earlyStoppingEpochs, dataTask=dataTask, informationText=informationText, pathCheckpoint=pathCheckpoint, 
                                                                                  verbose=verbose)
        elif (netType == "CNN" or netType == "TCN") and dataTask == "Regression":
            model, ownHistoryRetraining, finalTrainEpochsRetrain = CNN_TCN_Regression_Train_Loop(model=model, trainFeatures=targetTrainFeatures, trainLabels=targetTrainClasslabels,
                                                                                 checkFeatures=targetCheckFeatures, checkLabels=targetCheckClasslabels,
                                                                                 trainEpochs=trainEpochsRetrain, earlyStoppingYesNo=earlyStoppingYesNo, earlyStoppingEpochs=earlyStoppingEpochs,
                                                                                 featureType=featureType, netType=netType, metric=metric, informationText=informationText, pathCheckpoint=pathCheckpoint, verbose=verbose)
        elif (netType == "CNN" or netType == "TCN") and dataTask == "Classification":
            model, ownHistoryRetraining, finalTrainEpochsRetrain = CNN_TCN_Classification_Train_Loop(model=model, trainFeatures=targetTrainFeatures, trainLabels=targetTrainClasslabels,
                                                                                 checkFeatures=targetCheckFeatures, checkLabels=targetCheckClasslabels,
                                                                                 trainEpochs=trainEpochsRetrain, earlyStoppingYesNo=earlyStoppingYesNo, earlyStoppingEpochs=earlyStoppingEpochs, 
                                                                                 featureType=featureType, netType=netType, metric=metric, informationText=informationText, pathCheckpoint=pathCheckpoint, verbose=verbose)
        informationText = informationText.split(" - ")[0]
    else:
        finalTrainEpochsRetrain = None
    
    if fineTuningYesNo == True:
        informationText = informationText + ", Finetuning auf T  - Getestet auf T"  
        for layer in model.layers:
            layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fineTuningLearnRate), loss=lossFunction, metrics=[metric])
        if netType == "MLP":
            model, ownHistoryFinetuning, finalTrainEpochsFinetuning = MLP_Train_Loop(model=model, trainFeatures=targetTrainFeatures, trainLabels=targetTrainClasslabels,
                                                                                     checkFeatures=targetCheckFeatures, checkLabels=targetCheckClasslabels,
                                                                                     batchSize=batchSize, trainEpochs=trainEpochsFinetuning, earlyStoppingYesNo=earlyStoppingYesNo,
                                                                                     earlyStoppingEpochs=earlyStoppingEpochs, dataTask=dataTask, informationText=informationText, pathCheckpoint=pathCheckpoint,
                                                                                     verbose=verbose)
        elif (netType == "CNN" or netType == "TCN")   and dataTask == "Regression":
            model, ownHistoryFinetuning, finalTrainEpochsFinetuning = CNN_TCN_Regression_Train_Loop(model=model, trainFeatures=targetTrainFeatures, trainLabels=targetTrainClasslabels,
                                                                                    checkFeatures=targetCheckFeatures, checkLabels=targetCheckClasslabels,
                                                                                    trainEpochs=trainEpochsFinetuning, earlyStoppingYesNo=earlyStoppingYesNo, earlyStoppingEpochs=earlyStoppingEpochs,
                                                                                    featureType=featureType, netType=netType, metric=metric, informationText=informationText, pathCheckpoint=pathCheckpoint,
                                                                                    verbose=verbose)
        elif (netType == "CNN" or netType == "TCN") and dataTask == "Classification": 
            model, ownHistoryFinetuning, finalTrainEpochsFinetuning = CNN_TCN_Classification_Train_Loop(model=model, trainFeatures=targetTrainFeatures, trainLabels=targetTrainClasslabels,
                                                                                    checkFeatures=targetCheckFeatures, checkLabels=targetCheckClasslabels,
                                                                                    trainEpochs=trainEpochsFinetuning, earlyStoppingYesNo=earlyStoppingYesNo, earlyStoppingEpochs=earlyStoppingEpochs,
                                                                                    featureType=featureType, netType=netType, metric=metric, informationText=informationText, pathCheckpoint=pathCheckpoint,
                                                                                    verbose=verbose)
    else:
        finalTrainEpochsFinetuning = None
    
    informationText = informationText.split(" - ")[0]
    for layer in model.layers:
        layer.trainable = True
    
    return model, ownHistoryRetraining, ownHistoryFinetuning, finalTrainEpochsRetrain, finalTrainEpochsFinetuning, informationText


