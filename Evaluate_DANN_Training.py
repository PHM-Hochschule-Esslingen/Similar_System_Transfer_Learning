"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import tensorflow as tf
import numpy as np


def Evaluate_DANN_Net_During_Training(model, features, classLabels, domainLabels, learnRate, lambdaValue, lossFunction, metric, lossFunctionDomain, 
                                      metricDomain, tempTask, forceUseCodeClassification=False):
    
    DANN_model = model
    DANN_model_optimizer = tf.keras.optimizers.Adam(learning_rate=learnRate)
    DANN_model.compile(optimizer=DANN_model_optimizer,
                    loss={'Label_Output': lossFunction, 'Domain_Output': lossFunctionDomain},
                    loss_weights={'Label_Output': (1 - lambdaValue), 'Domain_Output': lambdaValue},
                    metrics={'Label_Output': [lossFunction, metric], 'Domain_Output': [lossFunctionDomain, metricDomain]})
    if tempTask == "Classification" or forceUseCodeClassification == True:
        Evaluation = []
        series_Val_Class_loss = []
        series_Val_Domain_loss = []
        series_Val_Class_metric = []
        series_Val_Domain_metric = []
        for i in range(features.shape[0]):
            x = np.expand_dims( features[i], axis=0 )
            y_Label = np.array([classLabels[i]])
            y_Label = y_Label.reshape(-1, 1) 
            y_Domain = np.array([domainLabels[i]])
            y_Domain = y_Domain.reshape(-1, 1)
            Evaluation.append( DANN_model.evaluate(x=x, y={'Label_Output': y_Label, 'Domain_Output': y_Domain}, verbose=0, return_dict=True) )
            testLoss = (1 - lambdaValue) * Evaluation[-1]["Label_Output_"+lossFunction] + lambdaValue * Evaluation[-1]["Domain_Output_"+lossFunctionDomain]
        for i in range( len(Evaluation) ):
            series_Val_Class_loss.append( Evaluation[i]["Label_Output"+"_"+lossFunction] )
            series_Val_Domain_loss.append( Evaluation[i]["Domain_Output"+"_"+lossFunctionDomain] )
            series_Val_Class_metric.append( Evaluation[i]["Label_Output"+"_"+metric] )
            series_Val_Domain_metric.append( Evaluation[i]["Domain_Output"+"_"+metricDomain] )
        series_Val_Class_loss =  np.nanmean(series_Val_Class_loss)
        series_Val_Domain_loss =  np.nanmean(series_Val_Domain_loss)
        series_Val_Class_metric =  np.nanmean(series_Val_Class_metric)
        series_Val_Domain_metric =  np.nanmean(series_Val_Domain_metric)
    if tempTask == "Regression" and forceUseCodeClassification == False:
        series_Val_Class_loss = []
        series_Val_Domain_loss = []
        series_Val_Class_metric = []
        series_Val_Domain_metric = []
        for i, (batch, batchLabel, batchDomain) in enumerate(zip(features, classLabels, domainLabels)):
            part_series_Val_Class_loss = []
            part_series_Val_Domain_loss = []
            part_series_Val_Class_metric = []
            part_series_Val_Domain_metric = []
            for countSubSeries in range(1, batch.shape[0]+1):
                subSeries = batch[:countSubSeries,:]
                subLabel = batchLabel[countSubSeries-1]
                subLabel = subLabel.reshape(-1,1)
                batchDomain = batchDomain.reshape(-1,1)
                evaluationResults = DANN_model.evaluate(x=np.array([subSeries]), y={'Label_Output':subLabel, 'Domain_Output':batchDomain} , verbose=0, return_dict=True)
                part_series_Val_Class_loss.append( evaluationResults["Label_Output"+"_"+lossFunction] )
                part_series_Val_Domain_loss.append( evaluationResults["Domain_Output"+"_"+lossFunctionDomain] )
                part_series_Val_Class_metric.append( evaluationResults["Label_Output"+"_"+metric] )
                part_series_Val_Domain_metric.append( evaluationResults["Domain_Output"+"_"+metricDomain] )
            series_Val_Class_loss.append( np.mean(part_series_Val_Class_loss) )
            series_Val_Domain_loss.append( np.mean(part_series_Val_Domain_loss) )
            series_Val_Class_metric.append( np.mean(part_series_Val_Class_metric) )
            series_Val_Domain_metric.append( np.mean(part_series_Val_Domain_metric) )
        series_Val_Class_loss =  np.nanmean(series_Val_Class_loss)
        series_Val_Domain_loss =  np.nanmean(series_Val_Domain_loss)
        series_Val_Class_metric =  np.nanmean(series_Val_Class_metric)
        series_Val_Domain_metric =  np.nanmean(series_Val_Domain_metric)
    EvaluationNames = ["Label_Output"+"_"+lossFunction, "Domain_Output"+"_"+lossFunctionDomain, "Label_Output"+"_"+metric, "Domain_Output"+"_"+metricDomain]
    return series_Val_Class_loss, series_Val_Domain_loss, series_Val_Class_metric, series_Val_Domain_metric, EvaluationNames
