"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import numpy as np
import pickle
from Fehlermetriken import Fehlerprognosemetriken, Klassifikationsfehlermetriken
from Helper_Functions import combineDics, calculateMean
from Data_Handling import Normalize_Data


def Evaluate_Regression_Nets(features, labels, featuresNames, normRulPy, normFeaturePy, model, informationText, featureType, activateDocu, path2):
    
    if featureType == "Points":
        loss = []
        metrics = []
        s = 1
        for i, (batch, batchLabel) in enumerate(zip(features, labels)):
            if activateDocu == True:
                y_pred = model.predict(batch, verbose=0)
                y_pred = np.squeeze(y_pred)
                y_pred_reNorm = Normalize_Data( y_pred, whatToDo="Rstand", norm=normRulPy)
                batchLabel_reNorn = Normalize_Data( batchLabel, whatToDo="Rstand", norm=normRulPy)
                normTime = {"mean": normFeaturePy["mean"][featuresNames == "Zeit"],
                            "std": normFeaturePy["std"][featuresNames == "Zeit"]}
                x_reNorm = Normalize_Data( batch[:,featuresNames == "Zeit"], whatToDo="Rstand", norm=normTime )
                metrics.append( Fehlerprognosemetriken(time=np.squeeze(x_reNorm, axis=(1,)),
                                       labels=batchLabel_reNorn,
                                       predictedLabels=y_pred_reNorm,
                                       alpha=0.1) )
                s = s + 1
            evaluationResults = model.evaluate(batch, batchLabel, verbose=0)
            for i in range(0, len(evaluationResults)):
                if model.metrics_names[i] == "loss":
                    loss.append(evaluationResults[i])
    elif featureType == "TimeSeries":
        s = 1
        loss = []
        metrics = []
        for i, (batch, batchLabel) in enumerate(zip(features, labels)):
            loss_part = []
            if activateDocu == True:
                y_pred = []
                y_reel = []
            for countSubSeries in range(1, batch.shape[0]+1): 
                subSeries = batch[:countSubSeries,:]
                subLabel = batchLabel[countSubSeries-1]
                if activateDocu == True:
                    y_pred.append( model.predict(np.array([subSeries]), verbose=0)[0] )
                    y_reel.append( subLabel )
                evaluationResults = model.evaluate(x=np.array([subSeries]), y=np.array([subLabel]), verbose=0)
                for i in range(0, len(evaluationResults)):
                    if model.metrics_names[i] == "loss":
                        loss_part.append(evaluationResults[i])
            if activateDocu == True:
                y_pred = np.array(y_pred)
                y_reel = np.array(y_reel)
            loss.append( np.mean(loss_part) )
            if activateDocu == True:
                y_pred_reNorm = Normalize_Data( y_pred, whatToDo="Rstand", norm=normRulPy)
                y_reel_reNorn = Normalize_Data( y_reel, whatToDo="Rstand", norm=normRulPy)
                normTime = {"mean": normFeaturePy["mean"][featuresNames == "Zeit"],
                    "std": normFeaturePy["std"][featuresNames == "Zeit"]}
                x_reNorm = Normalize_Data( batch[:,featuresNames == "Zeit"], whatToDo="Rstand", norm=normTime )
                s = s + 1
                metrics.append( Fehlerprognosemetriken(time=np.squeeze(x_reNorm, axis=(1,)),
                                       labels=y_reel_reNorn,
                                       predictedLabels = y_pred_reNorm,
                                       alpha=0.1) )
    loss = np.nanmean(loss)
    if activateDocu == True:
        metrics = combineDics(metrics)
        metrics = calculateMean(metrics)
        file = open(path2 + informationText +".pkl", "xb")
        pickle.dump(metrics, 
                    file)
        file.close()
    
    return loss


def Evaluate_Classification_Nets(model, features, labels, le, informationText, featureType, activateDocu, path2):
    
    if featureType == "Points":
        if activateDocu == True:
            predic_1 = np.argmax(model.predict(features, verbose=0),axis=-1)
        evaluationResults = model.evaluate(features, labels, verbose=0)
        for i in range(0, len(evaluationResults)):
            if model.metrics_names[i] == "loss":
                loss = evaluationResults[i]
    elif featureType == "TimeSeries":
        if activateDocu == True:
            predic_1 = []
        loss = []
        for batch, batchLabel in zip(features, labels):
            if activateDocu == True:
                predic_1.append( np.argmax( model.predict(np.array([batch]), verbose=0), axis=-1 ) )
            evaluationResults = model.evaluate(np.array([batch]), np.array([batchLabel]), verbose=0)
            for i in range(0, len(evaluationResults)):
                if model.metrics_names[i] == "loss":
                    loss.append(evaluationResults[i])
        loss = np.mean(loss)
    if activateDocu == True:
        metrics = Klassifikationsfehlermetriken(labels=labels, predictedLabels=predic_1)
        file = open(path2 + informationText +".pkl", "xb")
        pickle.dump(metrics, 
                    file)
        file.close()
    
    return loss

