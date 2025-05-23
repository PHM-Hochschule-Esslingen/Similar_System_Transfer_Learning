"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import pickle
import os
import numpy as np
import shutil
from pathlib import Path
import tensorflow as tf
from Coordination import Train_With_One_Source_Target_Split


def main():
    
    NET_TYPE = "MLP"
    
    SELECTED_DATA = "Filter"
    # SELECTED_DATA = "Bearing"
    
    FAIR_SPLIT_OPERATION_COND = True
    # FAIR_SPLIT_OPERATION_COND = False
    FAIR_SPLIT_FAULTS = True
    HOW_MANY_EXCLUDED_OPT_COND_TARGET_TRAIN_VAL = 3
    
    HYP_OPT_YES_NO = True
    HYP_OPT_STEPS = 150
    HYP_OPT_EARLY_STOPPINGE = 20
    
    RETRAIN_YES_NO = True
    FINE_TUNIG_YES_NO = True
    
    INITIAL_TRAINING_DATA = "Source"
    
    numberIterationsTrainValTestSplit = 10
    
    SOURCE_TEST_SIZE = 0.3
    TARGET_TEST_SIZE = 0.3
    SOURCE_VAL_SIZE = 0.2
    TARGET_VAL_SIZE = 0.2
    
    TRAIN_EPOCHS = 200
    TRAIN_EPOCHS_RETRAIN = 100
    TRAIN_EPOCHS_FINETUNING = 50
    EARLY_STOPPING_EPOCHS = 10
    
    directory = "Saved_Results/" + SELECTED_DATA + "_" + NET_TYPE
    specialTrainOps = ""
    if RETRAIN_YES_NO == False and FINE_TUNIG_YES_NO == False and (not "DANN" in NET_TYPE) :
        specialTrainOps = "_onlyInitTrain"
        if INITIAL_TRAINING_DATA == "Target":
            specialTrainOps = specialTrainOps + "_trainOnT"
        elif INITIAL_TRAINING_DATA == "Source":
            specialTrainOps = specialTrainOps + "_trainOnS"
        elif INITIAL_TRAINING_DATA == "All":
            specialTrainOps = specialTrainOps + "_trainOnA"
    directory = directory + specialTrainOps
    if HYP_OPT_YES_NO == True:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        myPath = Path(directory)
        myPath.mkdir(parents=True, exist_ok=True)
    else:
        delete_files_without_string(directory, "bestHyperparameters")
    
    allNetParameter = {}
    if "MLP" in NET_TYPE:
        allNetParameter.update({"featureType": "Points"})
    elif "CNN" in NET_TYPE or "TCN" in NET_TYPE:
        allNetParameter.update({"featureType": "TimeSeries"})
    if SELECTED_DATA == "Filter":
        DATASET = "featuresPy"
        dirname = os.path.dirname(__file__)
        FILE_PATH_ABSOLUT = os.path.join(dirname, 'Data/Filter/' + DATASET + ".pkl")
        with open(FILE_PATH_ABSOLUT, 'rb') as f: 
            Daten = pickle.load(f)
    elif SELECTED_DATA == "Bearing":
        DATASET = "features"
        if allNetParameter["featureType"] == "Points":
            DATASET = DATASET + "Py_classic" 
        elif allNetParameter["featureType"] == "TimeSeries":
            DATASET = DATASET + "Py_timeseries_mean" 
        dirname = os.path.dirname(__file__)
        FILE_PATH_ABSOLUT = os.path.join(dirname, "Data/Bearing/" + DATASET + ".pkl")
        with open(FILE_PATH_ABSOLUT, 'rb') as f: 
            Daten = pickle.load(f)
    allNetParameter.update({"ACTIVATION_DENSE": "relu"})
    allNetParameter.update({"ACTIVATION_CONV": "relu"})
    allNetParameter.update({"PADDING_CONV": "causal"})
    allNetParameter.update({"PADDING_POOL": "same"})
    
    if RETRAIN_YES_NO == False: TRAIN_EPOCHS_RETRAIN = None 
    if FINE_TUNIG_YES_NO == False: TRAIN_EPOCHS_FINETUNING = None
    
    if Daten["dataTask"] == "Regression":
        LOSS_FUNCTION = "MeanSquaredError"
        METRIC = "mae"
        LOSS_FUNCTION_DOMAIN = "BinaryCrossentropy"
        METRIC_DOMAIN = "binary_accuracy"
    elif Daten["dataTask"] == "Classification":
        LOSS_FUNCTION = "SparseCategoricalCrossentropy"
        METRIC = "sparse_categorical_accuracy"
        LOSS_FUNCTION_DOMAIN = "BinaryCrossentropy"
        METRIC_DOMAIN = "binary_accuracy"
    
    Parameter = {
        "NET_TYPE": NET_TYPE,
        "SOURCE_TEST_SIZE": SOURCE_TEST_SIZE, 
        "TARGET_TEST_SIZE": TARGET_TEST_SIZE,
        "SOURCE_VAL_SIZE": SOURCE_VAL_SIZE,
        "TARGET_VAL_SIZE": TARGET_VAL_SIZE,
        "RETRAIN_YES_NO": RETRAIN_YES_NO,
        "FINE_TUNIG_YES_NO": FINE_TUNIG_YES_NO,
        "EARLY_STOPPING_EPOCHS": EARLY_STOPPING_EPOCHS,
        "TRAIN_EPOCHS": TRAIN_EPOCHS,
        "TRAIN_EPOCHS_RETRAIN": TRAIN_EPOCHS_RETRAIN,
        "TRAIN_EPOCHS_FINETUNING": TRAIN_EPOCHS_FINETUNING,
        "HYP_OPT_YES_NO": HYP_OPT_YES_NO,
        "HYP_OPT_STEPS": HYP_OPT_STEPS,
        "HYP_OPT_EARLY_STOPPINGE": HYP_OPT_EARLY_STOPPINGE,
        "LOSS_FUNCTION": LOSS_FUNCTION,
        "METRIC": METRIC,
        "LOSS_FUNCTION_DOMAIN": LOSS_FUNCTION_DOMAIN,
        "METRIC_DOMAIN": METRIC_DOMAIN,
        "INITIAL_TRAINING_DATA": INITIAL_TRAINING_DATA,
        "FAIR_SPLIT_OPERATION_COND": FAIR_SPLIT_OPERATION_COND,
        "FAIR_SPLIT_FAULTS": FAIR_SPLIT_FAULTS,
        "HOW_MANY_EXCLUDED_OPT_COND_TARGET_TRAIN_VAL": HOW_MANY_EXCLUDED_OPT_COND_TARGET_TRAIN_VAL
    }
    Parameter.update(allNetParameter)
    del NET_TYPE, SOURCE_TEST_SIZE, SOURCE_VAL_SIZE, TARGET_VAL_SIZE, RETRAIN_YES_NO, FINE_TUNIG_YES_NO, EARLY_STOPPING_EPOCHS, TRAIN_EPOCHS, TRAIN_EPOCHS_RETRAIN, TRAIN_EPOCHS_FINETUNING,
    del HYP_OPT_YES_NO, HYP_OPT_STEPS, HYP_OPT_EARLY_STOPPINGE, LOSS_FUNCTION, METRIC, LOSS_FUNCTION_DOMAIN, METRIC_DOMAIN, INITIAL_TRAINING_DATA
    
    uniqueDomains = list(set(Daten["domainPy"]))
    tf.keras.utils.set_random_seed(1)
    randomSeeds = np.random.permutation(numberIterationsTrainValTestSplit)
    
    for TARGET_DOMAIN in uniqueDomains:
        pathWithTarget = directory + "/" + TARGET_DOMAIN
        myPath = Path(pathWithTarget)
        myPath.mkdir(parents=True, exist_ok=True)
    for TARGET_DOMAIN in uniqueDomains:
        for numRep in range(numberIterationsTrainValTestSplit):
            Train_With_One_Source_Target_Split(TARGET_DOMAIN=TARGET_DOMAIN, uniqueDomains=uniqueDomains, numberIterationsTrainValTestSplit=numberIterationsTrainValTestSplit, directory=directory,
                                               numRep=numRep, randomSeed=randomSeeds[numRep], Daten=Daten, Parameter=Parameter, task=Daten["dataTask"], NET_TYPE=Parameter["NET_TYPE"], SELECTED_DATA=Daten["data"], specialTrainOps=specialTrainOps,
                                               HYP_OPT_YES_NO=Parameter["HYP_OPT_YES_NO"])
    
    file = open(directory + "/" + "savedVariables_"+ Daten["data"] + "_" + Parameter["NET_TYPE"] + specialTrainOps + ".pkl", "xb")
    pickle.dump([Parameter, DATASET], 
                file)
    file.close()
    
    
def delete_files_without_string(root_folder, string):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if string not in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)

import fnmatch
def find_files_with_string(directory, string):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, f"*{string}*"):
            matches.append(filename)
    return matches




if __name__ == '__main__':
    main()
