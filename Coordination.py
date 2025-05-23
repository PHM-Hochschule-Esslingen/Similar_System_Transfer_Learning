"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import pickle
import numpy as np
from hyperopt import fmin, tpe, Trials
from hyperopt.early_stop import no_progress_loss
import tensorflow as tf
import copy

from Hyp_Opt import Hyp_Opt_Parameter_Space
from Nets_Collect import Get_Net
from Evaluate_Nets import Evaluate_Regression_Nets, Evaluate_Classification_Nets
from Retrain_Nets_TL import Retrain_Transfer_Learning
from Data_Handling import Devide_Data_And_Substitute_Labels, Normalize_Data


def Train_With_One_Source_Target_Split(TARGET_DOMAIN, uniqueDomains, numberIterationsTrainValTestSplit, directory, numRep, randomSeed, Daten, Parameter, task, NET_TYPE,
                                       SELECTED_DATA, specialTrainOps, HYP_OPT_YES_NO):
    
    print("\n------- Current Target: " + TARGET_DOMAIN + ", i.e. " + str(uniqueDomains.index(TARGET_DOMAIN)+1) + " of " + str(len(uniqueDomains)) + " -------", flush=True)
    pathWithTarget = directory + "/" + TARGET_DOMAIN
    print("\n----- Current Train/Val/Test: " + str(numRep+1) + " of " + str(numberIterationsTrainValTestSplit) + " -----", flush=True)
    tf.keras.utils.set_random_seed( int( randomSeed ))
    pathes = {}
    pathes["start"] = {
        "BEST_HYPERPARAMETERS": pathWithTarget + "/bestHyperparameters_"+ SELECTED_DATA + "_" + NET_TYPE + specialTrainOps + "_iter" + str(numRep) + ".pkl",
        "FINAL_EPOCHS": pathWithTarget + "/finalEpochs_"+ SELECTED_DATA + "_" + NET_TYPE + specialTrainOps + "_iter" + str(numRep) + ".pkl",
        "TEMP_CHECKPOINT_MODEL_EARLY_STOP": pathWithTarget + "/checkpointModel"+ "_iter" + str(numRep),
        "MODEL": pathWithTarget + "/",
        "FEHLERMETRIKEN": pathWithTarget + "/",
    }
    pathes["end"] = {
        "MODEL": "_model_" + SELECTED_DATA + "_" + NET_TYPE + specialTrainOps + "_iter" + str(numRep) + ".keras",
        "FEHLERMETRIKEN": "_fehlermetriken_" + SELECTED_DATA + "_" + NET_TYPE + specialTrainOps + "_iter" + str(numRep) + "_",
    }
    domains=Daten["domainPy"]
    nameOfTargetDataset=TARGET_DOMAIN
    if task == "Classification":
        optCond = Daten["operationSettingsPy"]
    else:
        optCond = None
    (sourceTrainFeatures, sourceTestFeatures, sourceValFeatures,
    sourceTrainClasslabels, sourceTestClasslabels, sourceValClasslabels,
    sourceTrainDomainlabels, sourceTestDomainlabels, sourceValDomainlabels,
    targetTrainFeatures, targetTestFeatures, targetValFeatures,
    targetTrainClasslabels, targetTestClasslabels, targetValClasslabels,
    targetTrainDomainlabels, targetTestDomainlabels, targetValDomainlabels,
    sourceTrainIdx, targetTrainIdx, sourceTestIdx, targetTestIdx, sourceValIdx, 
    targetValIdx, le) = Devide_Data_And_Substitute_Labels(features=Daten["signalPy"], labels=Daten["labelPy"], domains=domains, optCond=optCond,
                                                                  nameOfTargetDataset=nameOfTargetDataset, sourceTestSize=Parameter["SOURCE_TEST_SIZE"], targetTestSize=Parameter["TARGET_TEST_SIZE"], 
                                                                  sourceValSize=Parameter["SOURCE_VAL_SIZE"], targetValSize=Parameter["TARGET_VAL_SIZE"],
                                                                  task=Daten["dataTask"], featureType=Parameter["featureType"], fairSplitFaults=Parameter["FAIR_SPLIT_FAULTS"], 
                                                                  fairSplitOperationCond=Parameter["FAIR_SPLIT_OPERATION_COND"], howManyexcludedOptCondTargetTrainVal=Parameter["HOW_MANY_EXCLUDED_OPT_COND_TARGET_TRAIN_VAL"])
    if "DANN" in NET_TYPE and task == "Classification":
        sourceTrainOperation = Daten["operationSettingsPy"][sourceTrainIdx]
        targetTrainOperation = Daten["operationSettingsPy"][targetTrainIdx]
        sourceTestOperation = Daten["operationSettingsPy"][sourceTestIdx]
        targetTestOperation = Daten["operationSettingsPy"][targetTestIdx]
        sourceValOperation = Daten["operationSettingsPy"][sourceValIdx]
        targetValOperation = Daten["operationSettingsPy"][targetValIdx]
    Daten_devided = {
        "sourceTrainFeatures": copy.deepcopy(sourceTrainFeatures),
        "sourceTestFeatures": copy.deepcopy(sourceTestFeatures),
        "sourceValFeatures": copy.deepcopy(sourceValFeatures),
        "sourceTrainClasslabels": copy.deepcopy(sourceTrainClasslabels),
        "sourceTestClasslabels": copy.deepcopy(sourceTestClasslabels),
        "sourceValClasslabels": copy.deepcopy(sourceValClasslabels),
        "sourceTrainDomainlabels": copy.deepcopy(sourceTrainDomainlabels),
        "sourceTestDomainlabels": copy.deepcopy(sourceTestDomainlabels),
        "sourceValDomainlabels": copy.deepcopy(sourceValDomainlabels),
        "targetTrainFeatures": copy.deepcopy(targetTrainFeatures),
        "targetTestFeatures": copy.deepcopy(targetTestFeatures),
        "targetValFeatures": copy.deepcopy(targetValFeatures),
        "targetTrainClasslabels": copy.deepcopy(targetTrainClasslabels),
        "targetTestClasslabels": copy.deepcopy(targetTestClasslabels),
        "targetValClasslabels": copy.deepcopy(targetValClasslabels),
        "targetTrainDomainlabels": copy.deepcopy(targetTrainDomainlabels),
        "targetTestDomainlabels": copy.deepcopy(targetTestDomainlabels),
        "targetValDomainlabels": copy.deepcopy(targetValDomainlabels),
        "sourceTrainIdx": sourceTrainIdx, "targetTrainIdx": targetTrainIdx, "sourceTestIdx": sourceTestIdx,
        "targetTestIdx": targetTestIdx, "sourceValIdx": sourceValIdx, "targetValIdx": targetValIdx,
        "namePy": Daten["namePy"],
        "le": le,
        "signalNamesPy": Daten["signalNamesPy"],
        "data": Daten["data"],
        "dataTask" : Daten["dataTask"],
        "target": TARGET_DOMAIN
      }
    if "DANN" in NET_TYPE and task == "Classification":
        Daten_devided["sourceTrainOperation"] = copy.deepcopy(sourceTrainOperation)
        Daten_devided["targetTrainOperation"] = copy.deepcopy(targetTrainOperation)
        Daten_devided["sourceTestOperation"] = copy.deepcopy(sourceTestOperation)
        Daten_devided["targetTestOperation"] = copy.deepcopy(targetTestOperation)
        Daten_devided["sourceValOperation"] = copy.deepcopy(sourceValOperation)
        Daten_devided["targetValOperation"] = copy.deepcopy(targetValOperation)
    
    featuresForNorm = np.concatenate((Daten_devided["sourceTrainFeatures"], Daten_devided["targetTrainFeatures"]), axis=0)
    if Parameter["featureType"] == "TimeSeries" or task == "Regression":
        featuresForNorm = np.concatenate(featuresForNorm, axis=0)
    Daten_devided["normFeaturePy"] = Normalize_Data( featuresForNorm, whatToDo="standPara" )
    if task == "Regression":
        if "DANN" in NET_TYPE:
            labelsForNorm = Daten_devided["sourceTrainClasslabels"]
        else:
             labelsForNorm = np.concatenate((Daten_devided["sourceTrainClasslabels"], Daten_devided["targetTrainClasslabels"]), axis=0)
        labelsForNorm = np.concatenate(labelsForNorm, axis=0)
        Daten_devided["normRulPy"] = Normalize_Data( labelsForNorm, whatToDo="standPara" )  
    allFeatures = [s for s in list(Daten_devided.keys()) if "Features" in s]
    for iFeatures in allFeatures:
        if Parameter["featureType"] == "TimeSeries" or task == "Regression":
            for i in range(len(Daten_devided[iFeatures])):
                Daten_devided[iFeatures][i] = Normalize_Data( Daten_devided[iFeatures][i], whatToDo="stand", norm=Daten_devided["normFeaturePy"]  )
        else:
            Daten_devided[iFeatures] = Normalize_Data( Daten_devided[iFeatures], whatToDo="stand", norm=Daten_devided["normFeaturePy"]  )
    if task == "Regression":
        allLabels = [s for s in list(Daten_devided.keys()) if "Classlabels" in s]
        for iLabels in allLabels:
            for i in range(len(Daten_devided[iLabels])):
                Daten_devided[iLabels][i] = Normalize_Data( Daten_devided[iLabels][i], whatToDo="stand", norm=Daten_devided["normRulPy"] )
    Coordination_Function(Daten=Daten_devided, Parameter=Parameter, pathesParts=pathes, hypOptYesNo=HYP_OPT_YES_NO)
    file = open(pathWithTarget +  "/" + "savedVariables_"+ SELECTED_DATA + "_" + NET_TYPE + specialTrainOps + "_iter" + str(numRep) + ".pkl", "xb")
    pickle.dump([sourceTestIdx, targetTestIdx, sourceValIdx, targetValIdx, TARGET_DOMAIN], 
                file)
    file.close()


def Coordination_Function(Daten, Parameter, pathesParts, hypOptYesNo):
    
    hypOptSteps = Parameter["HYP_OPT_STEPS"]
    hypOptEarlyStopping=Parameter["HYP_OPT_EARLY_STOPPINGE"]
    netType = Parameter["NET_TYPE"]
    featureType = Parameter["featureType"]
    retrainYesNo = Parameter["RETRAIN_YES_NO"]
    fineTuningYesNo = Parameter["FINE_TUNIG_YES_NO"]
    activationDense = Parameter["ACTIVATION_DENSE"]
    activationConv = Parameter["ACTIVATION_CONV"]
    paddingConv = Parameter["PADDING_CONV"]
    paddingPool = Parameter["PADDING_POOL"]
    earlyStoppingEpochs = Parameter["EARLY_STOPPING_EPOCHS"]
    trainEpochs = Parameter["TRAIN_EPOCHS"]
    trainEpochsRetrain = Parameter["TRAIN_EPOCHS_RETRAIN"]
    trainEpochsFinetuning = Parameter["TRAIN_EPOCHS_FINETUNING"]
    lossFunction = Parameter["LOSS_FUNCTION"]
    metric = Parameter["METRIC"]
    lossFunctionDomain = Parameter["LOSS_FUNCTION_DOMAIN"]
    metricDomain = Parameter["METRIC_DOMAIN"]
    INITIAL_TRAINING_DATA = Parameter["INITIAL_TRAINING_DATA"]
    signalNamesPy = Daten["signalNamesPy"]
    normFeaturePy = Daten["normFeaturePy"]
    dataTask = Daten["dataTask"]
    data = Daten["data"]
    sourceTrainFeatures = Daten["sourceTrainFeatures"]
    sourceTestFeatures = Daten["sourceTestFeatures"]
    sourceValFeatures = Daten["sourceValFeatures"]
    sourceTrainClasslabels = Daten["sourceTrainClasslabels"]
    sourceTestClasslabels = Daten["sourceTestClasslabels"]
    sourceValClasslabels = Daten["sourceValClasslabels"]
    sourceTrainDomainlabels = Daten["sourceTrainDomainlabels"]
    sourceTestDomainlabels = Daten["sourceTestDomainlabels"]
    sourceValDomainlabels = Daten["sourceValDomainlabels"]
    targetTrainFeatures = Daten["targetTrainFeatures"]
    targetTestFeatures = Daten["targetTestFeatures"]
    targetValFeatures = Daten["targetValFeatures"]
    targetTrainClasslabels = Daten["targetTrainClasslabels"]
    targetTestClasslabels = Daten["targetTestClasslabels"]
    targetValClasslabels = Daten["targetValClasslabels"]
    targetTrainDomainlabels = Daten["targetTrainDomainlabels"]
    targetTestDomainlabels = Daten["targetTestDomainlabels"]
    targetValDomainlabels = Daten["targetValDomainlabels"]
    le = Daten["le"]
    
    if "DANN" in netType and dataTask == "Classification":
        sourceTrainOperation = Daten["sourceTrainOperation"]
        targetTrainOperation = Daten["targetTrainOperation"]
        sourceValOperation = Daten["sourceValOperation"]
        targetValOperation = Daten["targetValOperation"]
    else:
        sourceTrainOperation = None
        targetTrainOperation = None
        sourceValOperation = None
        targetValOperation = None
    if dataTask == "Regression":
        normRulPy = Daten["normRulPy"]
    elif dataTask == "Classification":
        normRulPy = []
    
    if hypOptYesNo == True:
        pathes = {}
        for keys in pathesParts["start"]:
            if keys in pathesParts["end"]:
                pathes[keys] = pathesParts["start"][keys] + "HYPOPT" + pathesParts["end"][keys]   
            else:
                pathes[keys] = pathesParts["start"][keys]
        hypPar_space = Hyp_Opt_Parameter_Space(netType, data)
        trials = Trials()
        best_hyperparameter = fmin(fn=lambda params: Training_And_Evaluation_Process_Coordination(currentParams=params, 
                                                                      sourceTrainFeatures=sourceTrainFeatures, sourceTrainClasslabels=sourceTrainClasslabels, sourceTrainDomainlabels=sourceTrainDomainlabels, sourceTrainOperation=sourceTrainOperation,
                                                                      sourceCheckFeatures=sourceValFeatures, sourceCheckClasslabels=sourceValClasslabels, sourceCheckDomainlabels=sourceValDomainlabels,
                                                                      targetTrainFeatures=targetTrainFeatures, targetTrainClasslabels=targetTrainClasslabels, targetTrainDomainlabels=targetTrainDomainlabels, targetTrainOperation=targetTrainOperation,
                                                                      targetCheckFeatures=targetValFeatures, targetCheckClasslabels=targetValClasslabels, targetCheckDomainlabels=targetValDomainlabels,
                                                                      featureType=featureType, dataTask=dataTask, netType=netType,verbose=False, pathes=pathes,
                                                                      activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool,
                                                                      lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                                                                      featuresNames=signalNamesPy, normRulPy=normRulPy, normFeaturePy=normFeaturePy, le=le, fineTuningYesNo=fineTuningYesNo, retrainYesNo=retrainYesNo,
                                                                      earlyStoppingYesNo=True, earlyStoppingEpochs=earlyStoppingEpochs, trainEpochs=trainEpochs, trainEpochsRetrain=trainEpochsRetrain, trainEpochsFinetuning=trainEpochsFinetuning,
                                                                      activateDocu=False, lossOrModel=True, INITIAL_TRAINING_DATA=INITIAL_TRAINING_DATA
                                                                      ),
                                    space=hypPar_space,
                                    algo=tpe.suggest,
                                    max_evals=hypOptSteps,
                                    trials=trials,
                                    early_stop_fn=no_progress_loss(hypOptEarlyStopping),
                                    verbose=True
                                    )
        
        for fruit in best_hyperparameter.keys():
           if best_hyperparameter[fruit] == int(best_hyperparameter[fruit]):
               best_hyperparameter[fruit] =  int(best_hyperparameter[fruit]) 
        file = open(pathes["BEST_HYPERPARAMETERS"], "xb")
        pickle.dump(best_hyperparameter, file)
        file.close()
    
    pathes = {}
    for keys in pathesParts["start"]:
        if keys in pathesParts["end"]:
            pathes[keys] = pathesParts["start"][keys] + "FINAL_FIND_EPOCHS" + pathesParts["end"][keys]
        else:
            pathes[keys] = pathesParts["start"][keys]
    file = open(pathes["BEST_HYPERPARAMETERS"], "rb")
    best_hyperparameter = pickle.load(file)
    file.close()
    model, finalEpochs = Training_And_Evaluation_Process_Coordination(currentParams=best_hyperparameter, 
                             sourceTrainFeatures=sourceTrainFeatures, sourceTrainClasslabels=sourceTrainClasslabels, sourceTrainDomainlabels=sourceTrainDomainlabels, sourceTrainOperation=sourceTrainOperation,
                             sourceCheckFeatures=sourceValFeatures, sourceCheckClasslabels=sourceValClasslabels, sourceCheckDomainlabels=sourceValDomainlabels,
                             targetTrainFeatures=targetTrainFeatures, targetTrainClasslabels=targetTrainClasslabels, targetTrainDomainlabels=targetTrainDomainlabels, targetTrainOperation=targetTrainOperation,
                             targetCheckFeatures=targetValFeatures, targetCheckClasslabels=targetValClasslabels, targetCheckDomainlabels=targetValDomainlabels,
                             featureType=featureType, dataTask=dataTask, netType=netType, verbose=False, pathes=pathes,
                             activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool,
                             lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                             featuresNames=signalNamesPy, normRulPy=normRulPy, normFeaturePy=normFeaturePy, le=le,  retrainYesNo=retrainYesNo, fineTuningYesNo=fineTuningYesNo,
                             earlyStoppingYesNo=True, earlyStoppingEpochs=earlyStoppingEpochs, trainEpochs=trainEpochs, trainEpochsRetrain=trainEpochsRetrain, trainEpochsFinetuning=trainEpochsFinetuning,
                             activateDocu=True, 
                             lossOrModel=False, INITIAL_TRAINING_DATA=INITIAL_TRAINING_DATA,
                             )
    finalTrainEpochs = finalEpochs["finalTrainEpochs"]
    finalTrainEpochsRetrain = None
    finalTrainEpochsFinetuning = None
    if (netType == "MLP" or netType == "CNN" or netType == "TCN"):
        if retrainYesNo == True: finalTrainEpochsRetrain = finalEpochs["finalTrainEpochsRetrain"]
        if fineTuningYesNo == True: finalTrainEpochsFinetuning = finalEpochs["finalTrainEpochsFinetuning"]
    
    pathes = {}
    for keys in pathesParts["start"]:
        if keys in pathesParts["end"]:
            pathes[keys] = pathesParts["start"][keys] + "FINAL" + pathesParts["end"][keys]
        else:
            pathes[keys] = pathesParts["start"][keys]
    sourceTrainValFeatures = np.concatenate((sourceTrainFeatures,sourceValFeatures))
    sourceTrainValClasslabels = np.concatenate([sourceTrainClasslabels, sourceValClasslabels])
    sourceTrainValDomainlabels = np.concatenate((sourceTrainDomainlabels, sourceValDomainlabels))
    targetTrainValFeatures = np.concatenate((targetTrainFeatures, targetValFeatures))
    targetTrainValClasslabels = np.concatenate((targetTrainClasslabels, targetValClasslabels))
    targetTrainValDomainlabels = np.concatenate((targetTrainDomainlabels, targetValDomainlabels))
    if "DANN" in netType and dataTask == "Classification":
        sourceTrainValOperation = np.concatenate((sourceTrainOperation, sourceValOperation))
        targetTrainValOperation = np.concatenate((targetTrainOperation, targetValOperation))
    else:
        sourceTrainValOperation = None
        targetTrainValOperation = None
    verboseFinal=True
    model, _ = Training_And_Evaluation_Process_Coordination(currentParams=best_hyperparameter, 
                    sourceTrainFeatures=sourceTrainValFeatures, sourceTrainClasslabels=sourceTrainValClasslabels, sourceTrainDomainlabels=sourceTrainValDomainlabels, sourceTrainOperation=sourceTrainValOperation,
                    sourceCheckFeatures=sourceTestFeatures, sourceCheckClasslabels=sourceTestClasslabels, sourceCheckDomainlabels=sourceTestDomainlabels,
                    targetTrainFeatures=targetTrainValFeatures, targetTrainClasslabels=targetTrainValClasslabels, targetTrainDomainlabels=targetTrainValDomainlabels, targetTrainOperation=targetTrainValOperation,
                    targetCheckFeatures=targetTestFeatures, targetCheckClasslabels=targetTestClasslabels, targetCheckDomainlabels=targetTestDomainlabels,
                    featureType=featureType, dataTask=dataTask, netType=netType, verbose=verboseFinal, pathes=pathes,
                    activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool,
                    lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                    featuresNames=signalNamesPy, normRulPy=normRulPy, normFeaturePy=normFeaturePy, le=le,  retrainYesNo=retrainYesNo, fineTuningYesNo=fineTuningYesNo,
                    earlyStoppingYesNo=False, earlyStoppingEpochs=[], trainEpochs=finalTrainEpochs, trainEpochsRetrain=finalTrainEpochsRetrain, trainEpochsFinetuning=finalTrainEpochsFinetuning,
                    activateDocu=True, 
                    lossOrModel=False, INITIAL_TRAINING_DATA=INITIAL_TRAINING_DATA,
                    )
    with open(pathes["FINAL_EPOCHS"], 'wb') as f:
        pickle.dump(finalEpochs, f)
    model.save(pathes["MODEL"])


def Training_And_Evaluation_Process_Coordination(currentParams, sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, sourceTrainOperation,
              sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels,
              targetTrainFeatures, targetTrainClasslabels, targetTrainDomainlabels, targetTrainOperation,
              targetCheckFeatures, targetCheckClasslabels, targetCheckDomainlabels,
              featureType, dataTask, netType, verbose, pathes,
              activationDense, activationConv, paddingConv, paddingPool,
              lossFunction, metric, lossFunctionDomain, metricDomain,
              featuresNames, normRulPy, normFeaturePy, le,  retrainYesNo, fineTuningYesNo,
              earlyStoppingYesNo, earlyStoppingEpochs, trainEpochs, trainEpochsRetrain, trainEpochsFinetuning,
              activateDocu, lossOrModel, INITIAL_TRAINING_DATA):
    
    if lossOrModel==False:
        model, finalEpochs = Training_And_Evaluation(currentParams, sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, sourceTrainOperation,
                      sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels,
                      targetTrainFeatures, targetTrainClasslabels, targetTrainDomainlabels, targetTrainOperation,
                      targetCheckFeatures, targetCheckClasslabels, targetCheckDomainlabels,
                      featureType, dataTask, netType, verbose, pathes,
                      activationDense, activationConv, paddingConv, paddingPool,
                      lossFunction, metric, lossFunctionDomain, metricDomain,
                      featuresNames, normRulPy, normFeaturePy, le,  retrainYesNo, fineTuningYesNo,
                      earlyStoppingYesNo, earlyStoppingEpochs, trainEpochs, trainEpochsRetrain, trainEpochsFinetuning,
                      activateDocu, lossOrModel, INITIAL_TRAINING_DATA)
    else:
        loss = Training_And_Evaluation(currentParams, sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, sourceTrainOperation,
                      sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels,
                      targetTrainFeatures, targetTrainClasslabels, targetTrainDomainlabels, targetTrainOperation,
                      targetCheckFeatures, targetCheckClasslabels, targetCheckDomainlabels,
                      featureType, dataTask, netType, verbose, pathes,
                      activationDense, activationConv, paddingConv, paddingPool,
                      lossFunction, metric, lossFunctionDomain, metricDomain,
                      featuresNames, normRulPy, normFeaturePy, le,  retrainYesNo, fineTuningYesNo,
                      earlyStoppingYesNo, earlyStoppingEpochs, trainEpochs, trainEpochsRetrain, trainEpochsFinetuning,
                      activateDocu, lossOrModel, INITIAL_TRAINING_DATA)
    if lossOrModel==False:
        return model, finalEpochs
    else:
        return loss


def Training_And_Evaluation(currentParams, sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, sourceTrainOperation,
              sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels,
              targetTrainFeatures, targetTrainClasslabels, targetTrainDomainlabels, targetTrainOperation,
              targetCheckFeatures, targetCheckClasslabels, targetCheckDomainlabels,
              featureType, dataTask, netType, verbose, pathes,
              activationDense, activationConv, paddingConv, paddingPool,
              lossFunction, metric, lossFunctionDomain, metricDomain,
              featuresNames, normRulPy, normFeaturePy, le,  retrainYesNo, fineTuningYesNo,
              earlyStoppingYesNo, earlyStoppingEpochs, trainEpochs, trainEpochsRetrain, trainEpochsFinetuning,
              activateDocu, lossOrModel, INITIAL_TRAINING_DATA):
    
    finalEpochs = {}
    model, finalEpochs["finalTrainEpochs"], informationText = Get_Net(sourceTrainFeatures=sourceTrainFeatures, sourceTrainClasslabels=sourceTrainClasslabels, sourceTrainDomainlabels=sourceTrainDomainlabels, sourceTrainOperation=sourceTrainOperation,
                          sourceCheckFeatures=sourceCheckFeatures, sourceCheckClasslabels=sourceCheckClasslabels, sourceCheckDomainlabels=sourceCheckDomainlabels,
                          targetTrainFeatures=targetTrainFeatures, targetTrainClasslabels=targetTrainClasslabels, targetTrainDomainlabels=targetTrainDomainlabels, targetTrainOperation=targetTrainOperation,
                          targetCheckFeatures=targetCheckFeatures, targetCheckClasslabels=targetCheckClasslabels, targetCheckDomainlabels=targetCheckDomainlabels,
                          featureType=featureType, dataTask=dataTask, netType=netType, verbose=verbose, activateDocu=activateDocu, pathes=pathes, 
                          hyperparameter=currentParams, activationDense=activationDense, activationConv=activationConv, paddingConv=paddingConv, paddingPool=paddingPool,
                          lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                          earlyStoppingYesNo=earlyStoppingYesNo, earlyStoppingEpochs=earlyStoppingEpochs, trainEpochs=trainEpochs, INITIAL_TRAINING_DATA=INITIAL_TRAINING_DATA)
    if "DANN" in netType:
        model_evaluate = tf.keras.Model(inputs=model.get_layer('Feature_Extractor_Input_Layer').output, outputs=[model.get_layer('Label_Output').output]) 
        optimizer = tf.keras.optimizers.Adam(learning_rate=currentParams["learnRate"])
        model_evaluate.compile(optimizer=optimizer, loss=lossFunction, weighted_metrics=[metric])
    else:
        model_evaluate = model
    if dataTask == "Regression":
        loss_S = Evaluate_Regression_Nets(features=sourceCheckFeatures, labels=sourceCheckClasslabels, featuresNames=featuresNames,
                                 normRulPy=normRulPy, normFeaturePy=normFeaturePy, model=model_evaluate,
                                 informationText=informationText + " - Getestet auf " + "S",
                                 featureType=featureType, activateDocu=activateDocu,
                                 path2=pathes["FEHLERMETRIKEN"] )
        loss_T = Evaluate_Regression_Nets(features=targetCheckFeatures, labels=targetCheckClasslabels, featuresNames=featuresNames,
                                 normRulPy=normRulPy, normFeaturePy=normFeaturePy, model=model_evaluate,
                                 informationText= informationText + " - Getestet auf T",
                                 featureType=featureType, activateDocu=activateDocu,
                                 path2=pathes["FEHLERMETRIKEN"] )
    elif dataTask == "Classification":
        loss_S = Evaluate_Classification_Nets(model=model_evaluate, features=sourceCheckFeatures, labels=sourceCheckClasslabels,
                                     le=le, informationText=informationText + " - Getestet auf " + "S",
                                     featureType=featureType, activateDocu=activateDocu,
                                     path2=pathes["FEHLERMETRIKEN"] )
        loss_T = Evaluate_Classification_Nets(model=model_evaluate, features=targetCheckFeatures, labels=targetCheckClasslabels,
                                     le=le, informationText=informationText + " - Getestet auf T",
                                     featureType=featureType, activateDocu=activateDocu,
                                     path2=pathes["FEHLERMETRIKEN"] )
    if "DANN" in netType:
        loss = loss_S
    elif INITIAL_TRAINING_DATA == "Source":
        loss = loss_S
    elif INITIAL_TRAINING_DATA == "Target":
        loss = loss_T
    elif INITIAL_TRAINING_DATA == "All":
        loss = loss_T
    
    if (netType == "MLP" or netType == "CNN" or netType == "TCN") and (retrainYesNo == True or fineTuningYesNo == True):
        lastLayersRetrainPercentage = currentParams["lastLayersRetrainPercentage"]
        learnRate = currentParams["learnRate"]
        learnRateFinetuning = currentParams["learnRateFinetuning"]
        if netType == "MLP":
            batchSize = currentParams["batchSize"]
        else:
            batchSize = []
        (model, ownHistoryRetraining, ownHistoryFinetuning, 
         finalEpochs["finalTrainEpochsRetrain"], 
         finalEpochs["finalTrainEpochsFinetuning"], informationText) = Retrain_Transfer_Learning(model=model, dataTask=dataTask, lastLayersRetrainPercentage=lastLayersRetrainPercentage, fineTuningYesNo=fineTuningYesNo, 
                                                                            retrainYesNo=retrainYesNo, verbose=verbose,
                                                                            targetTrainFeatures=targetTrainFeatures, targetTrainClasslabels=targetTrainClasslabels, targetCheckFeatures=targetCheckFeatures, targetCheckClasslabels=targetCheckClasslabels,
                                                                            trainEpochsRetrain=trainEpochsRetrain, batchSize=batchSize, netType=netType, learnRate=learnRate, fineTuningLearnRate=learnRateFinetuning, trainEpochsFinetuning=trainEpochsFinetuning,
                                                                            lossFunction=lossFunction, metric=metric, pathCheckpoint=pathes["TEMP_CHECKPOINT_MODEL_EARLY_STOP"], informationText=informationText,
                                                                            featureType=featureType, earlyStoppingYesNo=earlyStoppingYesNo, earlyStoppingEpochs=earlyStoppingEpochs)
        if dataTask == "Regression":
            _ = Evaluate_Regression_Nets(features=sourceCheckFeatures, labels=sourceCheckClasslabels, featuresNames=featuresNames,
                                     normRulPy=normRulPy, normFeaturePy=normFeaturePy, model=model,
                                     informationText=informationText + " - Getestet auf S",
                                     featureType=featureType, activateDocu=activateDocu,
                                     path2=pathes["FEHLERMETRIKEN"] )
            loss = Evaluate_Regression_Nets(features=targetCheckFeatures, labels=targetCheckClasslabels, featuresNames=featuresNames,
                                     normRulPy=normRulPy, normFeaturePy=normFeaturePy, model=model,
                                     informationText=informationText + " - Getestet auf T",
                                     featureType=featureType, activateDocu=activateDocu,
                                     path2=pathes["FEHLERMETRIKEN"] )
        elif dataTask == "Classification":
            _ = Evaluate_Classification_Nets(model=model, features=sourceCheckFeatures, labels=sourceCheckClasslabels,
                                         le=le, informationText= informationText + " - Getestet auf S",
                                         featureType=featureType, activateDocu=activateDocu,
                                         path2=pathes["FEHLERMETRIKEN"] )
            loss = Evaluate_Classification_Nets(model=model, features=targetCheckFeatures, labels=targetCheckClasslabels,
                                         le=le, informationText= informationText + " - Getestet auf T",
                                         featureType=featureType, activateDocu=activateDocu,
                                         path2=pathes["FEHLERMETRIKEN"] )
    if lossOrModel==False:
        return model, finalEpochs
    else:
        return loss