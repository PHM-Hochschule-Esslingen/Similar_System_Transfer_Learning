"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import tensorflow as tf
import numpy as np
import random
import os
import warnings

from Evaluate_DANN_Training import Evaluate_DANN_Net_During_Training
from Evaluate_Nets import Evaluate_Regression_Nets, Evaluate_Classification_Nets
from Data_Handling import Transform_Timeseries_To_Points_For_MLP
from Helper_Functions import combineDics, calculateMean


def MLP_Train_Loop(model, trainFeatures, trainLabels, checkFeatures, checkLabels, batchSize, trainEpochs, earlyStoppingYesNo, earlyStoppingEpochs, 
                    dataTask, informationText, pathCheckpoint, verbose):
    
    if earlyStoppingYesNo == True:
        checkpoint_filepath = pathCheckpoint + '.weights.h5'
    if dataTask == "Regression":
        trainFeatures, trainLabels = Transform_Timeseries_To_Points_For_MLP(features=trainFeatures, labels=trainLabels)
        checkFeatures, checkLabels = Transform_Timeseries_To_Points_For_MLP(features=checkFeatures, labels=checkLabels)
    
    startError= model.evaluate(checkFeatures, checkLabels, return_dict=True, batch_size=batchSize, verbose=False)
    startError = {f'val_{k}': v for k, v in startError.items()}
    
    if earlyStoppingYesNo == True:
        early_stopping = CustomEarlyStopping(monitor='val_loss', patience=earlyStoppingEpochs, restore_best_weights=False, initial_val_loss=startError["val_loss"])
        checkpoint = CustomModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor="val_loss", mode="min", save_best_only=True, initial_value_threshold=startError["val_loss"] , verbose=verbose)
        history = model.fit(trainFeatures, trainLabels, epochs=trainEpochs, batch_size=batchSize, validation_data=(checkFeatures, checkLabels), verbose=verbose, callbacks=[early_stopping, checkpoint])
        ownHistory = history.history
        for key in ownHistory:
            if "val" in key:
                ownHistory[key] = [ startError[key], *ownHistory[key] ]
            else:
                ownHistory[key] = [ float('NaN'), *ownHistory[key] ]
        finalEpochs = np.argmin(ownHistory["val_loss"]) 
        model.load_weights(checkpoint_filepath)
    else:
        history = model.fit(trainFeatures, trainLabels, epochs=trainEpochs, batch_size=batchSize, validation_data=(checkFeatures, checkLabels), verbose=verbose)
        ownHistory = history.history
        for key in ownHistory:
            if "val" in key:
                ownHistory[key] = [ startError[key], *ownHistory[key] ]
            else:
                ownHistory[key] = [ float('NaN'), *ownHistory[key] ]
        finalEpochs = trainEpochs
    
    if earlyStoppingYesNo == True:
        for key in ownHistory:
            ownHistory[key] = ownHistory[key][:finalEpochs+1]
        os.remove(checkpoint_filepath)
    
    return model, ownHistory, finalEpochs


def DANN_MLP_Train_Loop(model, sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, sourceTrainOperation, targetTrainFeatures, targetTrainDomainlabels, 
                        targetTrainOperation, sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels, targetCheckFeatures, targetCheckClasslabels, 
                        targetCheckDomainlabels, batchSize, trainEpochs, lambdaValue, learnRate, earlyStoppingYesNo, earlyStoppingEpochs, lossFunction, 
                        metric, lossFunctionDomain, metricDomain, dataTask, informationText, pathCheckpoint, verbose):
    
    if earlyStoppingYesNo == True:
        checkpoint_filepath = pathCheckpoint
    if dataTask == "Regression":
        sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels = Transform_Timeseries_To_Points_For_MLP(features=sourceTrainFeatures, labels=sourceTrainClasslabels, domainLabels=sourceTrainDomainlabels)
        targetTrainFeatures, targetTrainDomainlabels = Transform_Timeseries_To_Points_For_MLP(features=targetTrainFeatures, domainLabels=targetTrainDomainlabels)
        sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels = Transform_Timeseries_To_Points_For_MLP(features=sourceCheckFeatures, labels=sourceCheckClasslabels, domainLabels=sourceCheckDomainlabels)
        targetCheckFeatures, targetCheckClasslabels, targetCheckDomainlabels = Transform_Timeseries_To_Points_For_MLP(features=targetCheckFeatures, labels=targetCheckClasslabels, domainLabels=targetCheckDomainlabels)
    
    Feature_Extractor_Net = tf.keras.Model(inputs=model.get_layer('Feature_Extractor_Input_Layer').output, outputs=model.get_layer('Feature_Output').output)
    Label_Predictor_Net = tf.keras.Model(inputs=model.get_layer('Feature_Output').output, outputs=model.get_layer('Label_Output').output)
    Domain_Classifier_Net = tf.keras.Model(inputs=model.get_layer('Feature_Output').output, outputs=model.get_layer('Domain_Output').output)
    
    FE_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    LP_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    DC_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    
    if dataTask == "Classification":
        identifier = {"class_name": lossFunction,
                      "config": {"reduction": "sum_over_batch_size",
                                 "from_logits": False}}
        Label_lossfn = tf.keras.losses.get(identifier)
    elif dataTask == "Regression":
        identifier = {"class_name": lossFunction,
                      "config": {"reduction": "sum_over_batch_size"}}
        Label_lossfn = tf.keras.losses.get(identifier)
    
    identifier = {"class_name": lossFunctionDomain,
                  "config": {"reduction": "sum_over_batch_size",
                             "from_logits": False}}
    Domain_lossfn = tf.keras.losses.get(identifier)
    
    [startLossCheck, _, _, _, _] = Evaluate_DANN_Net_During_Training(model=model, features=sourceCheckFeatures, classLabels=sourceCheckClasslabels, domainLabels=sourceCheckDomainlabels,
                                            learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                                            tempTask=dataTask, forceUseCodeClassification=True)
    
    if earlyStoppingYesNo == True:
        early_stopper = EarlyStopper(patience=earlyStoppingEpochs, min_delta=0, startLoss=startLossCheck)
        finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
        finalEpochs = 0
    
    ownHistory = {}
    ownHistory["Training_Class_loss"] = []
    ownHistory["Training_Domain_loss"] = []
    ownHistory["Training_Total_loss"] = []
    ownHistory["Target_Check_Class_loss"] = []
    ownHistory["Target_Check_Domain_loss"] = []
    ownHistory["Target_Check_Class_metric"] = []
    ownHistory["Target_Check_Domain_metric"] = []
    ownHistory["Source_Check_Class_loss"] = []
    ownHistory["Source_Check_Domain_loss"] = []
    ownHistory["Source_Check_Class_metric"] = []
    ownHistory["Source_Check_Domain_metric"] = []
    
    if dataTask == "Regression":
        lengthes = [ len(sourceTrainDomainlabels), len(targetTrainDomainlabels) ]
        maxLen = max( lengthes )
        minLen = min( lengthes )
        maxLenIdx = lengthes.index(maxLen)
        if maxLen != minLen and maxLenIdx == 0:
            targetTrainFeatures = np.tile(targetTrainFeatures, (int(np.ceil(maxLen / minLen)), 1))[:maxLen]
            targetTrainDomainlabels = np.tile(targetTrainDomainlabels, int(np.ceil(maxLen / minLen)))[:maxLen]
        elif maxLen != minLen and maxLenIdx == 1:
            sourceTrainFeatures = np.tile(sourceTrainFeatures, (int(np.ceil(maxLen / minLen)), 1))[:maxLen]
            sourceTrainClasslabels = np.tile(sourceTrainClasslabels, int(np.ceil(maxLen / minLen)))[:maxLen]
            sourceTrainDomainlabels = np.tile(sourceTrainDomainlabels, int(np.ceil(maxLen / minLen)))[:maxLen]
    if dataTask == "Classification":
        uniqueStringsSource =  sorted(set(sourceTrainOperation))
        uniqueStringsTarget = sorted(set(targetTrainOperation))
        if uniqueStringsSource != uniqueStringsTarget:
            not_in_target = [item for item in uniqueStringsSource if item not in uniqueStringsTarget]
        else:
            not_in_target = []
    
    for epoch in range(trainEpochs):
        class_losses = []
        domain_losses = []
        total_losses = []
        
        if dataTask == "Regression":
            indicesSource = np.random.permutation(maxLen)
            indicesTarget = np.random.permutation(maxLen)
            combined_list = list(zip(indicesSource, indicesTarget))
            idxSourceTargetCombinations = np.array(combined_list)
        elif dataTask == "Classification":
            idxSourceTargetCombinations = []
            for curOpt in uniqueStringsSource:
                if curOpt not in not_in_target:
                    curSourceTrainIdx = np.where(sourceTrainOperation == curOpt)[0]
                    curTargetTrainIdx = np.where(targetTrainOperation == curOpt)[0]
                    if len(curSourceTrainIdx) > len(curTargetTrainIdx):
                        shorter_vector = curTargetTrainIdx
                        longer_vector = curSourceTrainIdx
                        is_source_longer = True
                    else:
                        shorter_vector = curSourceTrainIdx
                        longer_vector = curTargetTrainIdx
                        is_source_longer = False
                    zuordnung = []
                    shorter_vector_list = list(shorter_vector)
                    for v in longer_vector:
                        if not shorter_vector_list:
                            shorter_vector_list = list(shorter_vector)
                        element = random.choice(shorter_vector_list)
                        shorter_vector_list.remove(element)
                        if is_source_longer == True:
                            zuordnung.append((v, element))
                        else:
                            zuordnung.append((element, v))
                    idxSourceTargetCombinations = idxSourceTargetCombinations + zuordnung
                else:
                    curSourceTrainIdx = np.where(sourceTrainOperation == curOpt)[0]
                    curTargetTrainIdx = np.full(len(curSourceTrainIdx), np.nan)
                    zuordnung = list(zip(curSourceTrainIdx,curTargetTrainIdx))
                    idxSourceTargetCombinations = idxSourceTargetCombinations + zuordnung
            
            idxSourceTargetCombinations = np.array(idxSourceTargetCombinations)
            idxSourceTargetCombinations = np.random.permutation(idxSourceTargetCombinations)
        
        maxLen = idxSourceTargetCombinations.shape[0]
        for batchNumber in range(0, maxLen, batchSize):
            if maxLen >= batchNumber + batchSize:
                batch_indices = idxSourceTargetCombinations[batchNumber:batchNumber + batchSize, :]
            else:
                batch_indices = idxSourceTargetCombinations[batchNumber:, :]
            source_batch_indices_classification = batch_indices[:,0]
            batch_indices_adversarial = batch_indices[ np.where(~np.isnan(batch_indices[:,1]))[0], : ]
            source_batch_indices_adversarial = batch_indices_adversarial[:,0]
            target_batch_indices_adversarial = batch_indices_adversarial[:,1]
            
            if np.all(source_batch_indices_classification == source_batch_indices_classification.astype(int)):
                source_batch_indices_classification = source_batch_indices_classification.astype(int)
            if np.all(source_batch_indices_adversarial == source_batch_indices_adversarial.astype(int)):
                source_batch_indices_adversarial = source_batch_indices_adversarial.astype(int)
            if np.all(target_batch_indices_adversarial == target_batch_indices_adversarial.astype(int)):
                target_batch_indices_adversarial = target_batch_indices_adversarial.astype(int)
            
            source_batch_features_classification = sourceTrainFeatures[source_batch_indices_classification, :]
            source_batch_features_adversarial = sourceTrainFeatures[source_batch_indices_adversarial, :]
            source_batch_class_label_classification = sourceTrainClasslabels[source_batch_indices_classification]
            source_batch_domain_label_adversarial = sourceTrainDomainlabels[source_batch_indices_adversarial]
            target_batch_features_adversarial = targetTrainFeatures[target_batch_indices_adversarial, :]
            target_batch_domain_label_adversarial = targetTrainDomainlabels[target_batch_indices_adversarial]
            source_batch_class_label_classification_formated = tf.convert_to_tensor(source_batch_class_label_classification, dtype=tf.float32)
            
            with tf.GradientTape(persistent=True) as source_tape:
                Feature_Extractor_logits_s_classification = Feature_Extractor_Net(source_batch_features_classification)
                if batch_indices_adversarial.size != 0:
                    Feature_Extractor_logits_s_adversarial = Feature_Extractor_Net(source_batch_features_adversarial)
                    Feature_Extractor_logits_t_adversarial = Feature_Extractor_Net(target_batch_features_adversarial)
                class_classifier_logits_s = Label_Predictor_Net(Feature_Extractor_logits_s_classification)
                if batch_indices_adversarial.size != 0:
                    Domain_classifier_logits_s = Domain_Classifier_Net(Feature_Extractor_logits_s_adversarial)
                    Domain_classifier_logits_t = Domain_Classifier_Net(Feature_Extractor_logits_t_adversarial)
                if batch_indices_adversarial.size != 0:
                    Domain_classifier_logits_s_t = tf.concat([Domain_classifier_logits_s, Domain_classifier_logits_t], axis=0)
                    Domain_classifier_logits_s_t = Domain_classifier_logits_s_t[:,0]
                if batch_indices_adversarial.size != 0:
                    Domain_classifier_labels_s_t = tf.concat([source_batch_domain_label_adversarial, target_batch_domain_label_adversarial], axis=0)
                Class_classifier_loss_s = Label_lossfn(source_batch_class_label_classification_formated, class_classifier_logits_s)
                if batch_indices_adversarial.size != 0:
                    Domain_classifier_loss_s_t = Domain_lossfn(Domain_classifier_labels_s_t, Domain_classifier_logits_s_t)
                if batch_indices_adversarial.size != 0:
                    total_loss_s = (1 - lambdaValue) * Class_classifier_loss_s - lambdaValue * Domain_classifier_loss_s_t
                else:
                    total_loss_s = (1 - lambdaValue) * Class_classifier_loss_s
            
            Feature_Extractor_Net.trainable = False
            Label_Predictor_Net.trainable = True
            Domain_Classifier_Net.trainable = True
            Class_classifier_gradients_s = source_tape.gradient(Class_classifier_loss_s, Label_Predictor_Net.trainable_weights)
            if batch_indices_adversarial.size != 0:
                Domain_classifier_gradients_s_t = source_tape.gradient(Domain_classifier_loss_s_t, Domain_Classifier_Net.trainable_weights)
            LP_optimizer.apply_gradients(zip(Class_classifier_gradients_s, Label_Predictor_Net.trainable_weights))
            if batch_indices_adversarial.size != 0:
                DC_optimizer.apply_gradients(zip(Domain_classifier_gradients_s_t, Domain_Classifier_Net.trainable_weights))
            Feature_Extractor_Net.trainable = True
            Label_Predictor_Net.trainable = False
            Domain_Classifier_Net.trainable = False
            Feature_extractor_gradients_s = source_tape.gradient(total_loss_s, Feature_Extractor_Net.trainable_weights)
            FE_optimizer.apply_gradients(zip(Feature_extractor_gradients_s, Feature_Extractor_Net.trainable_weights))
            class_losses.append(Class_classifier_loss_s.numpy())
            if batch_indices_adversarial.size != 0:
                domain_losses.append(Domain_classifier_loss_s_t.numpy())
            total_losses.append(total_loss_s.numpy())

        avg_class_loss = sum(class_losses) / len(class_losses)
        if batch_indices_adversarial.size != 0:
            avg_domain_loss = sum(domain_losses) / len(domain_losses)
        else:
            avg_domain_loss = np.NaN
        avg_total_loss = sum(total_losses) / len(total_losses)
        
        ownHistory["Training_Class_loss"].append(avg_class_loss)
        ownHistory["Training_Domain_loss"].append(avg_domain_loss)
        ownHistory["Training_Total_loss"].append(avg_total_loss)
        
        [series_Check_Class_loss, series_Check_Domain_loss, 
        series_Check_Class_metric, series_Check_Domain_metric, 
        EvaluationNames] = Evaluate_DANN_Net_During_Training(model=model, features=targetCheckFeatures, classLabels=targetCheckClasslabels, domainLabels=targetCheckDomainlabels,
                                                             learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain,
                                                             metricDomain=metricDomain, tempTask="Classification", forceUseCodeClassification=True)
        
        ownHistory["Target_Check_Class_loss"].append( np.mean(series_Check_Class_loss) )
        ownHistory["Target_Check_Domain_loss"].append( np.mean(series_Check_Domain_loss) )
        ownHistory["Target_Check_Class_metric"].append( np.mean(series_Check_Class_metric) )
        ownHistory["Target_Check_Domain_metric"].append( np.mean(series_Check_Domain_metric) )
        
        [series_Check_Class_loss, series_Check_Domain_loss,
        series_Check_Class_metric, series_Check_Domain_metric,
        EvaluationNames] = Evaluate_DANN_Net_During_Training(model=model, features=sourceCheckFeatures, classLabels=sourceCheckClasslabels, domainLabels=sourceCheckDomainlabels,
                                                             learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain,
                                                             metricDomain=metricDomain, tempTask="Classification", forceUseCodeClassification=True)
        
        ownHistory["Source_Check_Class_loss"].append( np.mean(series_Check_Class_loss) )
        ownHistory["Source_Check_Domain_loss"].append( np.mean(series_Check_Domain_loss) )
        ownHistory["Source_Check_Class_metric"].append( np.mean(series_Check_Class_metric) )
        ownHistory["Source_Check_Domain_metric"].append( np.mean(series_Check_Domain_metric) )
        
        if earlyStoppingYesNo == True:
            breakEarly, saveCurrModel = early_stopper.early_stop(ownHistory["Source_Check_Class_loss"][-1], verbose)
            if saveCurrModel == True:
                finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
                finalEpochs = epoch + 1
            if breakEarly == True:
                break
    
    if earlyStoppingYesNo == True: 
        model = finalModel
    
    for key in ownHistory:
        if key == "Source_Check_Class_loss":
            ownHistory[key] = [ startLossCheck, *ownHistory[key] ]
        else:
            ownHistory[key] = [ float('NaN'), *ownHistory[key] ]
    
    if earlyStoppingYesNo == True: 
        for key in ownHistory:
            ownHistory[key] = ownHistory[key][:finalEpochs+1]
    else:
        finalEpochs = trainEpochs
    
    return model, ownHistory, finalEpochs


def CNN_TCN_Regression_Train_Loop(model, trainFeatures, trainLabels, checkFeatures, checkLabels, trainEpochs, earlyStoppingYesNo, earlyStoppingEpochs,
                                  featureType, netType, metric, informationText, pathCheckpoint, verbose):
    
    if earlyStoppingYesNo == True:
        checkpoint_filepath = pathCheckpoint
    
    startLossCheck = Evaluate_Regression_Nets(features=checkFeatures, labels=checkLabels, featuresNames=[], normRulPy=[], normFeaturePy=[], model=model, 
                                              informationText=informationText, featureType=featureType, activateDocu=False, path2=[]) 
    
    if earlyStoppingYesNo == True:
        early_stopper = EarlyStopper(patience=earlyStoppingEpochs, min_delta=0, startLoss=startLossCheck)
        finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
        finalEpochs = 0
    
    ownHistory = []
    lossCheck = []
    for Epochs in range(0, trainEpochs):
        meanHistoryList = []
        for i, (batch, RUL_batch) in enumerate(zip(trainFeatures, trainLabels)):
            historyList = []
            for countSubSeries in range(1, batch.shape[0]+1):
                subSeries = batch[:countSubSeries,:]
                subLabel = RUL_batch[countSubSeries-1]
                history = model.fit(np.array([subSeries]), np.array([subLabel]), epochs=1, verbose=0)
                historyList.append(history.history)
            
            tempVal = combineDics(historyList)
            meanHistoryList.append( calculateMean(tempVal) )
        
        tempVal = combineDics(meanHistoryList)
        ownHistory.append( calculateMean(tempVal) )
        
        lossCheck.append( Evaluate_Regression_Nets(features=checkFeatures, labels=checkLabels, featuresNames=[], normRulPy=[], normFeaturePy=[], model=model, 
                                                   informationText=informationText, featureType=featureType, activateDocu=False, path2=[]) )
        
        if earlyStoppingYesNo == True:
            breakEarly, saveCurrModel = early_stopper.early_stop(lossCheck[-1], verbose)
            if saveCurrModel == True:
                finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
                finalEpochs = Epochs + 1
            if breakEarly == True:
                break
    
    if earlyStoppingYesNo == True: 
        model = finalModel 
    
    ownHistory = combineDics(ownHistory)
    ownHistory["val_loss"] = lossCheck
    
    for key in ownHistory:
        if key == "val_loss":
            ownHistory[key] = [ startLossCheck, *ownHistory[key] ]
        else:
            ownHistory[key] = [ float('NaN'), *ownHistory[key] ]
    
    if earlyStoppingYesNo == True: 
        for key in ownHistory:
            ownHistory[key] = ownHistory[key][:finalEpochs+1]
    else:
        finalEpochs = trainEpochs
    
    return model, ownHistory, finalEpochs


def DANN_Regression_Training_Loop(model, sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, targetTrainFeatures, targetTrainDomainlabels, 
                                  sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels, targetCheckFeatures, targetCheckClasslabels, 
                                  targetCheckDomainlabels, trainEpochs, lambdaValue, learnRate, earlyStoppingYesNo, earlyStoppingEpochs, lossFunction, 
                                  metric, lossFunctionDomain, metricDomain, informationText, pathCheckpoint, verbose):
    
    if earlyStoppingYesNo == True:
        checkpoint_filepath = pathCheckpoint
    
    Feature_Extractor_Net = tf.keras.Model(inputs=model.get_layer('Feature_Extractor_Input_Layer').output, outputs=model.get_layer('Feature_Output').output)
    Label_Predictor_Net = tf.keras.Model(inputs=model.get_layer('Feature_Output').output, outputs=model.get_layer('Label_Output').output)
    Domain_Classifier_Net = tf.keras.Model(inputs=model.get_layer('Feature_Output').output, outputs=model.get_layer('Domain_Output').output)
    
    FE_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    LP_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    DC_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    
    Label_lossfn = tf.keras.losses.get(lossFunction)
    identifier = {"class_name": lossFunctionDomain,
                  "config": {"reduction": "sum_over_batch_size",
                             "from_logits": False}}
    Domain_lossfn = tf.keras.losses.get(identifier)
    
    [startLossCheck, _, _, _, _] = Evaluate_DANN_Net_During_Training(model=model, features=sourceCheckFeatures, classLabels=sourceCheckClasslabels, domainLabels=sourceCheckDomainlabels,
                                            learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                                            tempTask="Regression")
    
    if earlyStoppingYesNo == True:
        early_stopper = EarlyStopper(patience=earlyStoppingEpochs, min_delta=0, startLoss=startLossCheck)
        finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
        finalEpochs = 0
    
    ownHistory = {}
    ownHistory["Training_Class_loss"] = []
    ownHistory["Training_Domain_loss"] = []
    ownHistory["Training_Total_loss"] = []
    ownHistory["Target_Check_Class_loss"] = []
    ownHistory["Target_Check_Domain_loss"] = []
    ownHistory["Target_Check_Class_metric"] = []
    ownHistory["Target_Check_Domain_metric"] = []
    ownHistory["Source_Check_Class_loss"] = []
    ownHistory["Source_Check_Domain_loss"] = []
    ownHistory["Source_Check_Class_metric"] = []
    ownHistory["Source_Check_Domain_metric"] = []
    
    lengthes = [ len(sourceTrainDomainlabels), len(targetTrainDomainlabels) ]
    maxLen = max( lengthes )
    minLen = min( lengthes )
    maxLenIdx = lengthes.index(maxLen)
    
    if maxLenIdx == 0:
        targetTrainFeatures = np.tile(targetTrainFeatures, (int(np.ceil(maxLen / minLen))))[:maxLen]
        targetTrainDomainlabels = np.tile(targetTrainDomainlabels, int(np.ceil(maxLen / minLen)))[:maxLen]
    elif maxLenIdx == 1:
        sourceTrainFeatures = np.tile(sourceTrainFeatures, (int(np.ceil(maxLen / minLen))))[:maxLen]
        sourceTrainClasslabels = np.tile(sourceTrainClasslabels, int(np.ceil(maxLen / minLen)))[:maxLen]
        sourceTrainDomainlabels = np.tile(sourceTrainDomainlabels, int(np.ceil(maxLen / minLen)))[:maxLen]
    
    for epoch in range(trainEpochs):
        class_loss_batches = []
        domain_loss_batches = []
        total_loss_batches = []
        
        indices = np.random.permutation(maxLen)
        sourceTrainFeatures = sourceTrainFeatures[indices]
        sourceTrainClasslabels = sourceTrainClasslabels[indices]
        sourceTrainDomainlabels = sourceTrainDomainlabels[indices]
        indices = np.random.permutation(maxLen)
        targetTrainFeatures = targetTrainFeatures[indices]
        targetTrainDomainlabels = targetTrainDomainlabels[indices]
        
        for i, (source_batch, source_RUL_label_batch, source_domain_label_batch, target_batch, target_domain_label_batch) in enumerate(zip(sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, targetTrainFeatures, targetTrainDomainlabels)):
            source_batch_formated = source_batch
            target_batch_formated = target_batch
            source_label_batch_formated = source_RUL_label_batch
            subSourceDomain = np.array([source_domain_label_batch])
            subTargetDomain = np.array([target_domain_label_batch])
            maxSeriesIdx = np.argmax([source_batch_formated.shape[0], target_batch_formated.shape[0]])
            maxSeriesVal = np.max([source_batch_formated.shape[0], target_batch_formated.shape[0]])
            minSeriesVal = np.min([source_batch_formated.shape[0], target_batch_formated.shape[0]])
            
            class_loss_timeSeries = []
            domain_loss_timeSeries = []
            total_loss_timeSeries = []
            countSubSeriesSmall = 1
            
            for countSubSeries in range(1, maxSeriesVal+1):
                if maxSeriesIdx == 0:
                    subSourceSeries = np.expand_dims( source_batch_formated[:countSubSeries,:], axis=0)
                    subTargetSeries = np.expand_dims( target_batch_formated[:countSubSeriesSmall,:], axis=0)
                    subSourceLabel = source_label_batch_formated[countSubSeries-1]
                else:
                    subSourceSeries = np.expand_dims( source_batch_formated[:countSubSeriesSmall,:], axis=0)
                    subTargetSeries = np.expand_dims( target_batch_formated[:countSubSeries,:], axis=0)
                    subSourceLabel = source_label_batch_formated[countSubSeriesSmall-1]
                
                if countSubSeriesSmall % minSeriesVal == 0:
                    countSubSeriesSmall = 1
                else:
                    countSubSeriesSmall = countSubSeriesSmall + 1
                
                with tf.GradientTape(persistent=True) as source_tape:
                    Feature_Extractor_logits_s = Feature_Extractor_Net(subSourceSeries)
                    Feature_Extractor_logits_t = Feature_Extractor_Net(subTargetSeries)
                    Regression_RUL_logits_s = Label_Predictor_Net(Feature_Extractor_logits_s)
                    Domain_classifier_logits_s = Domain_Classifier_Net(Feature_Extractor_logits_s)
                    Domain_classifier_logits_t = Domain_Classifier_Net(Feature_Extractor_logits_t)
                    Domain_classifier_logits_s_t = tf.concat([Domain_classifier_logits_s, Domain_classifier_logits_t], axis=0)
                    Domain_classifier_logits_s_t = tf.reshape(Domain_classifier_logits_s_t, [-1])
                    Domain_classifier_labels_s_t = tf.concat([subSourceDomain, subTargetDomain], axis=0)
                    Regression_RUL_loss_s = Label_lossfn(subSourceLabel, Regression_RUL_logits_s)
                    Domain_classifier_loss_s_t = Domain_lossfn(Domain_classifier_labels_s_t, Domain_classifier_logits_s_t)
                    total_loss_s = (1 - lambdaValue) * Regression_RUL_loss_s - lambdaValue * Domain_classifier_loss_s_t
                
                Feature_Extractor_Net.trainable = False
                Label_Predictor_Net.trainable = True
                Domain_Classifier_Net.trainable = True
                 
                Regression_RUL_gradients_s = source_tape.gradient(Regression_RUL_loss_s, Label_Predictor_Net.trainable_weights)
                Domain_classifier_gradients_s_t = source_tape.gradient(Domain_classifier_loss_s_t, Domain_Classifier_Net.trainable_weights)
                LP_optimizer.apply_gradients(zip(Regression_RUL_gradients_s, Label_Predictor_Net.trainable_weights))
                DC_optimizer.apply_gradients(zip(Domain_classifier_gradients_s_t, Domain_Classifier_Net.trainable_weights))
                
                Feature_Extractor_Net.trainable = True
                Label_Predictor_Net.trainable = False
                Domain_Classifier_Net.trainable = False
                
                Feature_extractor_gradients_s = source_tape.gradient(total_loss_s, Feature_Extractor_Net.trainable_weights)
                FE_optimizer.apply_gradients(zip(Feature_extractor_gradients_s, Feature_Extractor_Net.trainable_weights))
                
                class_loss_timeSeries.append(Regression_RUL_loss_s.numpy())
                domain_loss_timeSeries.append(Domain_classifier_loss_s_t.numpy())
                total_loss_timeSeries.append(total_loss_s.numpy())
            
            class_loss_batches.append( np.mean(class_loss_timeSeries) )
            domain_loss_batches.append( np.mean(domain_loss_timeSeries) )
            total_loss_batches.append( np.mean(total_loss_timeSeries) )
        
        avg_RUL_loss = np.nanmean(class_loss_batches)
        avg_domain_loss = np.nanmean(domain_loss_batches)
        avg_total_loss = np.nanmean(total_loss_batches)
        
        ownHistory["Training_Class_loss"].append(avg_RUL_loss)
        ownHistory["Training_Domain_loss"].append(avg_domain_loss)
        ownHistory["Training_Total_loss"].append(avg_total_loss)
        
        # Netz evaluieren auf Target Validierungsdaten
        [series_Check_Class_loss, series_Check_Domain_loss, 
        series_Check_Class_metric, series_Check_Domain_metric, 
        EvaluationNames] = Evaluate_DANN_Net_During_Training(model=model, features=targetCheckFeatures, classLabels=targetCheckClasslabels, domainLabels=targetCheckDomainlabels,
                                                            learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain,
                                                            metricDomain=metricDomain, tempTask="Regression")
               
        ownHistory["Target_Check_Class_loss"].append( np.mean(series_Check_Class_loss) )
        ownHistory["Target_Check_Domain_loss"].append( np.mean(series_Check_Domain_loss) )
        ownHistory["Target_Check_Class_metric"].append( np.mean(series_Check_Class_metric) )
        ownHistory["Target_Check_Domain_metric"].append( np.mean(series_Check_Domain_metric) )
        
        [series_Check_Class_loss, series_Check_Domain_loss, 
        series_Check_Class_metric, series_Check_Domain_metric, 
        EvaluationNames] = Evaluate_DANN_Net_During_Training(model=model, features=sourceCheckFeatures, classLabels=sourceCheckClasslabels, domainLabels=sourceCheckDomainlabels,
                                                            learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, 
                                                            metricDomain=metricDomain, tempTask="Regression")
        
        ownHistory["Source_Check_Class_loss"].append( np.mean(series_Check_Class_loss) )
        ownHistory["Source_Check_Domain_loss"].append( np.mean(series_Check_Domain_loss) )
        ownHistory["Source_Check_Class_metric"].append( np.mean(series_Check_Class_metric) )
        ownHistory["Source_Check_Domain_metric"].append( np.mean(series_Check_Domain_metric) )
        
        if earlyStoppingYesNo == True:
            breakEarly, saveCurrModel = early_stopper.early_stop(ownHistory["Source_Check_Class_loss"][-1], verbose)
            if saveCurrModel == True:
                finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
                finalEpochs = epoch + 1
            if breakEarly == True:
                break
    
    if earlyStoppingYesNo == True:
        model = finalModel
    
    for key in ownHistory:
        if key == "Source_Check_Class_loss":
            ownHistory[key] = [ startLossCheck, *ownHistory[key] ]
        else:
            ownHistory[key] = [ float('NaN'), *ownHistory[key] ]
    
    if earlyStoppingYesNo == True: 
        for key in ownHistory:
            ownHistory[key] = ownHistory[key][:finalEpochs+1]
    else:
        finalEpochs = trainEpochs
    
    return model, ownHistory, finalEpochs


def CNN_TCN_Classification_Train_Loop(model, trainFeatures, trainLabels, checkFeatures, checkLabels, trainEpochs, earlyStoppingYesNo, earlyStoppingEpochs, 
                                      featureType, netType, metric, informationText, pathCheckpoint, verbose):
    
    if earlyStoppingYesNo == True:
        checkpoint_filepath = pathCheckpoint
    
    Counter_Variable = 1
    
    startLossCheck = Evaluate_Classification_Nets(model=model, features=checkFeatures, labels=checkLabels, le=[],  informationText=informationText, featureType=featureType, 
                                                  activateDocu=False, path2=[]) 
    
    if earlyStoppingYesNo == True:
        early_stopper = EarlyStopper(patience=earlyStoppingEpochs, min_delta=0, startLoss=startLossCheck)
        finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
        finalEpochs = 0
    
    ownHistory = []
    lossCheck = []
    for Epochs in range(0, trainEpochs):
        historyList = []
        for batch, labels_batch in zip(trainFeatures, trainLabels):
            history = model.fit(np.array([batch]), np.array([labels_batch]), epochs=1, verbose=0)
            historyList.append(history.history)
            Counter_Variable = Counter_Variable + 1
         
        tempVal = combineDics(historyList)
        ownHistory.append( calculateMean(tempVal) )
        
        lossCheck.append( Evaluate_Classification_Nets(model=model, features=checkFeatures, labels=checkLabels, le=[], informationText=informationText, featureType=featureType,
                        activateDocu=False, path2=[]) )
        
        if earlyStoppingYesNo == True:
            breakEarly, saveCurrModel = early_stopper.early_stop(lossCheck[-1], verbose)
            if saveCurrModel == True:
                finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
                finalEpochs = Epochs + 1
            if breakEarly == True:
                break
    
    if earlyStoppingYesNo == True: 
        model = finalModel 
    
    ownHistory = combineDics(ownHistory)
    ownHistory["val_loss"] = lossCheck
    
    for key in ownHistory:
        if key == "val_loss":
            ownHistory[key] = [ startLossCheck, *ownHistory[key] ]
        else:
            ownHistory[key] = [ float('NaN'), *ownHistory[key] ]
    
    if earlyStoppingYesNo == True: 
        for key in ownHistory:
            ownHistory[key] = ownHistory[key][:finalEpochs+1]
    else:
        finalEpochs = trainEpochs
    
    return model, ownHistory, finalEpochs

#@tf.function(reduce_retracing=True)
def DANN_Classification_Training_Loop(model, sourceTrainFeatures, sourceTrainClasslabels, sourceTrainDomainlabels, sourceTrainOperation, targetTrainFeatures, targetTrainDomainlabels, 
                                      targetTrainOperation, sourceCheckFeatures, sourceCheckClasslabels, sourceCheckDomainlabels, targetCheckFeatures, targetCheckClasslabels, 
                                      targetCheckDomainlabels, trainEpochs, lambdaValue, learnRate, earlyStoppingYesNo, earlyStoppingEpochs, lossFunction, 
                                      metric, lossFunctionDomain, metricDomain, informationText, pathCheckpoint, verbose):
    
    if earlyStoppingYesNo == True:
        checkpoint_filepath = pathCheckpoint
    
    Feature_Extractor_Net = tf.keras.Model(inputs=model.get_layer('Feature_Extractor_Input_Layer').output, outputs=model.get_layer('Feature_Output').output)
    Label_Predictor_Net = tf.keras.Model(inputs=model.get_layer('Feature_Output').output, outputs=model.get_layer('Label_Output').output)
    Domain_Classifier_Net = tf.keras.Model(inputs=model.get_layer('Feature_Output').output, outputs=model.get_layer('Domain_Output').output)
    
    FE_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    LP_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    DC_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    
    identifier = {"class_name": lossFunction,
                  "config": {"reduction": "sum_over_batch_size",
                             "from_logits": False}}
    Label_lossfn = tf.keras.losses.get(identifier)
    
    identifier = {"class_name": lossFunctionDomain,
                  "config": {"reduction": "sum_over_batch_size",
                             "from_logits": False}}
    Domain_lossfn = tf.keras.losses.get(identifier)
    
    [startLossCheck, _, _, _, _] = Evaluate_DANN_Net_During_Training(model=model, features=sourceCheckFeatures, classLabels=sourceCheckClasslabels, domainLabels=sourceCheckDomainlabels,
                                            learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, metricDomain=metricDomain,
                                            tempTask="Classification")
     
    if earlyStoppingYesNo == True:
        early_stopper = EarlyStopper(patience=earlyStoppingEpochs, min_delta=0, startLoss=startLossCheck)
        finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
        finalEpochs = 0
    
    ownHistory = {}
    ownHistory["Training_Class_loss"] = []
    ownHistory["Training_Domain_loss"] = []
    ownHistory["Training_Total_loss"] = []
    ownHistory["Target_Check_Class_loss"] = []
    ownHistory["Target_Check_Domain_loss"] = []
    ownHistory["Target_Check_Class_metric"] = []
    ownHistory["Target_Check_Domain_metric"] = []
    ownHistory["Source_Check_Class_loss"] = []
    ownHistory["Source_Check_Domain_loss"] = []
    ownHistory["Source_Check_Class_metric"] = []
    ownHistory["Source_Check_Domain_metric"] = []
    
    uniqueStringsSource =  sorted(set(sourceTrainOperation))
    uniqueStringsTarget = sorted(set(targetTrainOperation))
    if uniqueStringsSource != uniqueStringsTarget:
        not_in_target = [item for item in uniqueStringsSource if item not in uniqueStringsTarget]
    else:
        not_in_target = []

    for epoch in range(trainEpochs):
        class_losses = []
        domain_losses = []
        total_losses = []
        idxSourceTargetCombinations = []
        
        for curOpt in uniqueStringsSource:
            if curOpt not in not_in_target:
                curSourceTrainIdx = np.where(sourceTrainOperation == curOpt)[0]
                curTargetTrainIdx = np.where(targetTrainOperation == curOpt)[0]

                if len(curSourceTrainIdx) > len(curTargetTrainIdx):
                    shorter_vector = curTargetTrainIdx
                    longer_vector = curSourceTrainIdx
                    is_source_longer = True
                else:
                    shorter_vector = curSourceTrainIdx
                    longer_vector = curTargetTrainIdx
                    is_source_longer = False
                
                zuordnung = []
                shorter_vector_list = list(shorter_vector)
                for v in longer_vector:
                    if not shorter_vector_list:
                        shorter_vector_list = list(shorter_vector)
                    element = random.choice(shorter_vector_list)
                    shorter_vector_list.remove(element)
                    if is_source_longer == True:
                        zuordnung.append((v, element))
                    else:
                        zuordnung.append((element, v))
                idxSourceTargetCombinations = idxSourceTargetCombinations + zuordnung
            else:
                curSourceTrainIdx = np.where(sourceTrainOperation == curOpt)[0]
                curTargetTrainIdx = np.full(len(curSourceTrainIdx), np.nan)
                zuordnung = list(zip(curSourceTrainIdx,curTargetTrainIdx))
                idxSourceTargetCombinations = idxSourceTargetCombinations + zuordnung
        
        idxSourceTargetCombinations = np.array(idxSourceTargetCombinations)
        idxSourceTargetCombinations = np.random.permutation(idxSourceTargetCombinations)
        
        for sourceIdx, targetIdx in idxSourceTargetCombinations:
            
            if sourceIdx.is_integer():
                sourceIdx = int(sourceIdx)
            if targetIdx.is_integer():
                targetIdx = int(targetIdx)
            
            source_batch = sourceTrainFeatures[sourceIdx]
            source_class_label_batch = sourceTrainClasslabels[sourceIdx]
            source_domain_label_batch = sourceTrainDomainlabels[sourceIdx]
            source_batch_formated = np.expand_dims(source_batch, axis=0)
            source_class_label_batch_formated = tf.convert_to_tensor([source_class_label_batch], dtype=tf.float32)
            
            if ~np.isnan(targetIdx):
                target_batch = targetTrainFeatures[targetIdx]
                target_domain_label_batch = targetTrainDomainlabels[targetIdx]
                target_batch_formated = np.expand_dims(target_batch, axis=0)
            else:
                target_batch = None
                target_domain_label_batch = None
            
            with tf.GradientTape(persistent=True) as source_tape:
                Feature_Extractor_logits_s = Feature_Extractor_Net(source_batch_formated)
                if ~np.isnan(targetIdx):
                    Feature_Extractor_logits_t = Feature_Extractor_Net(target_batch_formated)
                class_classifier_logits_s = Label_Predictor_Net(Feature_Extractor_logits_s)
                if ~np.isnan(targetIdx):
                    Domain_classifier_logits_s = Domain_Classifier_Net(Feature_Extractor_logits_s)
                    Domain_classifier_logits_t = Domain_Classifier_Net(Feature_Extractor_logits_t)
                if ~np.isnan(targetIdx):
                    Domain_classifier_logits_s_t = tf.concat([Domain_classifier_logits_s, Domain_classifier_logits_t], axis=0)
                if ~np.isnan(targetIdx):
                    Domain_classifier_labels_s_t = tf.convert_to_tensor([np.array([source_domain_label_batch]), np.array([target_domain_label_batch])])
                Class_classifier_loss_s = Label_lossfn(source_class_label_batch_formated, class_classifier_logits_s)
                if ~np.isnan(targetIdx):
                    Domain_classifier_loss_s_t = Domain_lossfn(tf.reshape(Domain_classifier_labels_s_t, (Domain_classifier_labels_s_t.shape[0],)), tf.reshape(Domain_classifier_logits_s_t, (Domain_classifier_logits_s_t.shape[0],)))
                if ~np.isnan(targetIdx):
                    total_loss_s = (1 - lambdaValue) * Class_classifier_loss_s - lambdaValue * Domain_classifier_loss_s_t
                else:
                    total_loss_s = (1 - lambdaValue) * Class_classifier_loss_s
            
            Feature_Extractor_Net.trainable = False
            Label_Predictor_Net.trainable = True
            Domain_Classifier_Net.trainable = True
            
            Class_classifier_gradients_s = source_tape.gradient(Class_classifier_loss_s, Label_Predictor_Net.trainable_weights)
            if ~np.isnan(targetIdx):
                Domain_classifier_gradients_s_t = source_tape.gradient(Domain_classifier_loss_s_t, Domain_Classifier_Net.trainable_weights)
            
            LP_optimizer.apply_gradients(zip(Class_classifier_gradients_s, Label_Predictor_Net.trainable_weights))
            if ~np.isnan(targetIdx):
                DC_optimizer.apply_gradients(zip(Domain_classifier_gradients_s_t, Domain_Classifier_Net.trainable_weights))
            
            Feature_Extractor_Net.trainable = True
            Label_Predictor_Net.trainable = False
            Domain_Classifier_Net.trainable = False
            
            Feature_extractor_gradients_s = source_tape.gradient(total_loss_s, Feature_Extractor_Net.trainable_weights)
            FE_optimizer.apply_gradients(zip(Feature_extractor_gradients_s, Feature_Extractor_Net.trainable_weights))
            class_losses.append(Class_classifier_loss_s.numpy())
            if ~np.isnan(targetIdx):
                domain_losses.append(Domain_classifier_loss_s_t.numpy())
            total_losses.append(total_loss_s.numpy())
        
        avg_class_loss = sum(class_losses) / len(class_losses)
        if domain_losses != []:
            avg_domain_loss = sum(domain_losses) / len(domain_losses)
        else:
            avg_domain_loss = np.NaN
        avg_total_loss = sum(total_losses) / len(total_losses)
        
        ownHistory["Training_Class_loss"].append(avg_class_loss)
        ownHistory["Training_Domain_loss"].append(avg_domain_loss)
        ownHistory["Training_Total_loss"].append(avg_total_loss)
        
        # Netz evaluieren auf Target Validierungsdaten
        [series_Check_Class_loss, series_Check_Domain_loss, 
        series_Check_Class_metric, series_Check_Domain_metric, 
        EvaluationNames] = Evaluate_DANN_Net_During_Training(model=model, features=targetCheckFeatures, classLabels=targetCheckClasslabels, domainLabels=targetCheckDomainlabels,
                                                             learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain,
                                                             metricDomain=metricDomain, tempTask="Classification")
        
        ownHistory["Target_Check_Class_loss"].append( np.mean(series_Check_Class_loss) )
        ownHistory["Target_Check_Domain_loss"].append( np.mean(series_Check_Domain_loss) )
        ownHistory["Target_Check_Class_metric"].append( np.mean(series_Check_Class_metric) )
        ownHistory["Target_Check_Domain_metric"].append( np.mean(series_Check_Domain_metric) )
        
        [series_Check_Class_loss, series_Check_Domain_loss, 
        series_Check_Class_metric, series_Check_Domain_metric, 
        EvaluationNames] = Evaluate_DANN_Net_During_Training(model=model, features=sourceCheckFeatures, classLabels=sourceCheckClasslabels, domainLabels=sourceCheckDomainlabels,
                                                             learnRate=learnRate, lambdaValue=lambdaValue, lossFunction=lossFunction, metric=metric, lossFunctionDomain=lossFunctionDomain, 
                                                             metricDomain=metricDomain, tempTask="Classification")
        
        ownHistory["Source_Check_Class_loss"].append( np.mean(series_Check_Class_loss) )
        ownHistory["Source_Check_Domain_loss"].append( np.mean(series_Check_Domain_loss) )
        ownHistory["Source_Check_Class_metric"].append( np.mean(series_Check_Class_metric) )
        ownHistory["Source_Check_Domain_metric"].append( np.mean(series_Check_Domain_metric) )
        
        if earlyStoppingYesNo == True:
            breakEarly, saveCurrModel = early_stopper.early_stop(ownHistory["Source_Check_Class_loss"][-1], verbose)
            if saveCurrModel == True:
                finalModel = SaveAndStoreNet(model=model, metric=metric, checkpoint_filepath=checkpoint_filepath)
                finalEpochs = epoch + 1
            if breakEarly == True:
                break
    
    if earlyStoppingYesNo == True: 
        model = finalModel
    
    for key in ownHistory:
        if key == "Source_Check_Class_loss":
            ownHistory[key] = [ startLossCheck, *ownHistory[key] ]
        else:
            ownHistory[key] = [ float('NaN'), *ownHistory[key] ]
    
    if earlyStoppingYesNo == True:
        for key in ownHistory:
            ownHistory[key] = ownHistory[key][:finalEpochs+1]
    else:
        finalEpochs = trainEpochs
    
    return model, ownHistory, finalEpochs


def SaveAndStoreNet(model, metric, checkpoint_filepath):
    
    model_json = model.to_json()
    with open(checkpoint_filepath+".json", 'w') as file:
        file.write(model_json)
    model.save_weights(checkpoint_filepath+ ".weights.h5")
    with open(checkpoint_filepath+".json", "r") as file: 
        finalModel = file.read()
    finalModel = tf.keras.models.model_from_json(finalModel)
    lossFunction = model.loss
    optimizer_config = model.optimizer.get_config()
    new_optimizer = type(model.optimizer).from_config(optimizer_config)
    finalModel.compile(optimizer=new_optimizer, loss=lossFunction, weighted_metrics=[metric])
    finalModel.load_weights(checkpoint_filepath+ ".weights.h5")
    os.remove(checkpoint_filepath+ ".weights.h5")
    os.remove(checkpoint_filepath+".json")
    
    return finalModel


class EarlyStopper:
    
    def __init__(self, patience=1, min_delta=0, startLoss=[]):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        if startLoss == []:
            self.min_validation_loss = float('inf')
        else:
            self.min_validation_loss = startLoss
    
    def early_stop(self, validation_loss, verbose):
        breakEarly = False
        saveCurrModel = False
        if validation_loss < self.min_validation_loss:
            if verbose == True:
                print("Verbesserung des Val-Losses (Bisher bester Loss: " + str(self.min_validation_loss) + " , neuer Loss: " + str(validation_loss) + ")")
            self.min_validation_loss = validation_loss
            self.counter = 0
            saveCurrModel = True
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            if verbose == True:
                print("KEINE Verbesserung des Val-Losses (Bisher bester Loss: " + str(self.min_validation_loss) + " , neuer Loss: " + str(validation_loss) + ")")
            self.counter += 1
            if self.counter >= self.patience:
                breakEarly = True
                if verbose == True:
                    print("Early Stopping aktiviert, nach " + str(self.counter) + " Epochen mit schlechterem Abschneiden")
        return breakEarly, saveCurrModel


from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.trainers import compile_utils
from keras.src.utils import io_utils

@keras_export("keras.callbacks.EarlyStopping")
class CustomEarlyStopping(Callback):

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
        initial_val_loss=None
        
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_epoch = start_from_epoch
        self.initial_val_loss = initial_val_loss
        self.best = initial_val_loss

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"EarlyStopping mode {mode} is unknown, fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"
        self.mode = mode
        self.monitor_op = None

    def _set_monitor_op(self):
        if self.mode == "min":
            self.monitor_op = ops.less
        elif self.mode == "max":
            self.monitor_op = ops.greater
        else:
            metric_name = self.monitor.removeprefix("val_")
            if metric_name == "loss":
                self.monitor_op = ops.less
            if hasattr(self.model, "metrics"):
                all_metrics = []
                for m in self.model.metrics:
                    if isinstance(
                        m,
                        (
                            compile_utils.CompileMetrics,
                            compile_utils.MetricsList,
                        ),
                    ):
                        all_metrics.extend(m.metrics)
                for m in all_metrics:
                    if m.name == metric_name:
                        if hasattr(m, "_direction"):
                            if m._direction == "up":
                                self.monitor_op = ops.greater
                            else:
                                self.monitor_op = ops.less
        if self.monitor_op is None:
            raise ValueError(
                f"EarlyStopping callback received monitor={self.monitor} "
                "but Keras isn't able to automatically determine whether "
                "that metric should be maximized or minimized. "
                "Pass `mode='max'` in order to do early stopping based "
                "on the highest metric value, or pass `mode='min'` "
                "in order to use the lowest value."
            )
        if self.monitor_op == ops.less:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op is None:
            self._set_monitor_op()

        current = self.get_monitor_value(logs)
        if current is None or epoch < self.start_from_epoch:
            return
        if self.restore_best_weights and self.best_weights is None:
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0
            return

        if self.wait >= self.patience and epoch >= 0:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )
        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                io_utils.print_msg(
                    "Restoring model weights from "
                    "the end of the best epoch: "
                    f"{self.best_epoch + 1}."
                )
            self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                (
                    f"Early stopping conditioned on metric `{self.monitor}` "
                    "which is not available. "
                    f"Available metrics are: {','.join(list(logs.keys()))}"
                ),
                stacklevel=2,
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)


import re
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import file_utils


@keras_export("keras.callbacks.ModelCheckpoint")
class CustomModelCheckpoint(Callback):

    def __init__(
        self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    ):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = file_utils.path_to_string(filepath)
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self.best = initial_value_threshold

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"ModelCheckpoint mode '{mode}' is unknown, "
                "fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.Inf
            else:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.Inf

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                "Expected save_freq are 'epoch' or integer values"
            )

        if save_weights_only:
            if not self.filepath.endswith(".weights.h5"):
                raise ValueError(
                    "When using `save_weights_only=True` in `ModelCheckpoint`"
                    ", the filepath provided must end in `.weights.h5` "
                    "(Keras weights format). Received: "
                    f"filepath={self.filepath}"
                )
        else:
            if not self.filepath.endswith(".keras"):
                raise ValueError(
                    "The filepath provided must end in `.keras` "
                    "(Keras model format). Received: "
                    f"filepath={self.filepath}"
                )

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
        
        if epoch == 0:
            filepath = self._get_file_path(epoch=epoch-1, batch=None, logs=logs)
            self.model.save_weights(filepath, overwrite=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == "epoch":
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == "epoch":
            return False
        if batch <= self._last_batch_seen:
            add_batches = batch + 1
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _save_model(self, epoch, batch, logs):
        """Saves the model.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        filepath = self._get_file_path(epoch, batch, logs)
        dirname = os.path.dirname(filepath)
        if dirname and not file_utils.exists(dirname):
            file_utils.makedirs(dirname)

        try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        f"Can save best model only with {self.monitor} "
                        "available, skipping.",
                        stacklevel=2,
                    )
                elif (
                    isinstance(current, np.ndarray)
                    or backend.is_tensor(current)
                ) and len(current.shape) > 0:
                    warnings.warn(
                        "Can save best model only when `monitor` is "
                        f"a scalar value. Received: {current}. "
                        "Falling back to `save_best_only=False`."
                    )
                    self.model.save(filepath, overwrite=True)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch + 1}: {self.monitor} "
                                "improved "
                                f"from {self.best:.5f} to {current:.5f}, "
                                f"saving model to {filepath}"
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch + 1}: "
                                f"{self.monitor} did not improve "
                                f"from {self.best:.5f}"
                            )
            else:
                if self.verbose > 0:
                    io_utils.print_msg(
                        f"\nEpoch {epoch + 1}: saving model to {filepath}"
                    )
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
        except IsADirectoryError:  # h5py 3.x
            raise IOError(
                "Please specify a non-directory filepath for "
                "ModelCheckpoint. Filepath used is an existing "
                f"directory: {filepath}"
            )
        except IOError as e:
            if "is a directory" in str(e.args[0]).lower():
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: f{filepath}"
                )
            raise e

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""

        try:
            if batch is None or "batch" in logs:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(
                    epoch=epoch + 1, batch=batch + 1, **logs
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". '
                f"Reason: {e}"
            )
        return file_path

    def _checkpoint_exists(self, filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        return file_utils.exists(filepath)

    def _get_most_recently_modified_file_matching_pattern(self, pattern):

        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = "^" + re.sub(r"{.*}", r".*", base_name) + "$"

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if file_utils.exists(dir_name):
            for file_name in os.listdir(dir_name):
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (
                        file_path_with_largest_file_name is None
                        or file_path > file_path_with_largest_file_name
                    ):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        n_file_with_latest_mod_time += 1
        
        if n_file_with_latest_mod_time == 1:
            return file_path_with_latest_mod_time
        else:
            return file_path_with_largest_file_name




