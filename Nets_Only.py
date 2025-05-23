"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, Input
from tensorflow.keras.models import Sequential 
from tcn import TCN


def MLP_Net(InputLayer, OutputLayer, hyperparameter, activationDense, dataTask):
    
    Modell = Sequential()
    Modell.add(Input(shape=(InputLayer,)))
    numberHidLayer = hyperparameter["numberHidLayer"]
    if numberHidLayer== int(numberHidLayer):
        numberHidLayer =  int(numberHidLayer)
    for i in range(0, numberHidLayer):
        layerSize = hyperparameter["layerSize" + str(i)]
        dropoutRate = hyperparameter["dropoutRate" + str(i)]
        Modell.add(Dense(layerSize, activation=activationDense))
        Modell.add(Dropout(dropoutRate))
    if dataTask == "Regression":
        Modell.add(Dense(OutputLayer, activation='linear'))
    elif dataTask == "Classification":
        Modell.add(Dense(OutputLayer, activation='softmax'))
    return Modell

def CNN_Net(InputLayer, OutputLayer, hyperparameter, activationDense, activationConv, paddingConv, paddingPool, dataTask):
    
    Modell = Sequential()
    numberHidLayerConv = hyperparameter["numberHidLayerConv"]
    if numberHidLayerConv== int(numberHidLayerConv):
        numberHidLayerConv =  int(numberHidLayerConv)
    Modell.add(Input(shape=(None, InputLayer)))
    if numberHidLayerConv < 2:
        Modell.add(Conv1D(hyperparameter["numFiltersConv0"], kernel_size=(hyperparameter["filterSizeConv0"]), strides=1, padding=paddingConv, activation=activationConv))
        Modell.add(GlobalMaxPooling1D())
    else:
        for i in range(0, numberHidLayerConv-1):
            numFiltersConv = hyperparameter["numFiltersConv" + str(i)]
            filterSizeConv = hyperparameter["filterSizeConv" + str(i)]
            poolSizeConv = hyperparameter["poolSizeConv" + str(i)]
            Modell.add(Conv1D(numFiltersConv, kernel_size=(filterSizeConv), strides=1, padding=paddingConv, activation=activationConv))
            Modell.add(MaxPooling1D(pool_size=poolSizeConv, strides=poolSizeConv, padding=paddingPool))
        Modell.add(Conv1D(hyperparameter["numFiltersConv" + str(numberHidLayerConv-1)], kernel_size=(hyperparameter["filterSizeConv" + str(numberHidLayerConv-1)]), strides=1, padding=paddingConv, activation=activationConv))
        Modell.add(GlobalMaxPooling1D())
    numberHidLayerDense = hyperparameter["numberHidLayerDense"]
    if numberHidLayerDense== int(numberHidLayerDense):
        numberHidLayerDense =  int(numberHidLayerDense)
    for i in range(0, numberHidLayerDense):
        layerSizeDense = hyperparameter["layerSizeDense" + str(i)]
        dropoutRateDense = hyperparameter["dropoutRateDense" + str(i)]
        Modell.add(Dense(layerSizeDense, activation=activationDense))
        Modell.add(Dropout(dropoutRateDense))
    if dataTask == "Regression":
        Modell.add(Dense(OutputLayer, activation='linear'))
    elif dataTask == "Classification":
        Modell.add(Dense(OutputLayer, activation='softmax'))
    return Modell

def TCN_Net(InputLayer, OutputLayer, hyperparameter, activationDense, activationConv, paddingConv, paddingPool, dataTask):
    
    dilations = []
    for i in range(0,hyperparameter["numberLayerConvInBlock"]):
        dilations.append(2**i)
    Modell = Sequential()
    numberBlocks = hyperparameter["numberBlocks"]
    if numberBlocks== int(numberBlocks):
        numberBlocks =  int(numberBlocks)
    Modell.add(Input(shape=(None, InputLayer)))
    if numberBlocks < 2:
        Modell.add( TCN(nb_filters=hyperparameter["numFiltersConv0"], kernel_size=(hyperparameter["filterSizeConv0"]), nb_stacks=1, dilations=dilations, padding=paddingConv, use_skip_connections=True, dropout_rate=0.0, return_sequences=True,
                activation=activationConv) )
        Modell.add(GlobalMaxPooling1D())
    else:
        for i in range(0, numberBlocks-1):
            numFiltersConv = hyperparameter["numFiltersConv" + str(i)]
            filterSizeConv = hyperparameter["filterSizeConv" + str(i)]
            poolSizeConv = hyperparameter["poolSizeConv" + str(i)]
            Modell.add( TCN(nb_filters=numFiltersConv, kernel_size=(filterSizeConv), nb_stacks=1, dilations=dilations, padding=paddingConv, use_skip_connections=True, dropout_rate=0.0, return_sequences=True,
                    activation=activationConv) )
            Modell.add(MaxPooling1D(pool_size=poolSizeConv, strides=poolSizeConv, padding=paddingPool))
        Modell.add( TCN(nb_filters=hyperparameter["numFiltersConv" + str(numberBlocks-1)], kernel_size=(hyperparameter["filterSizeConv" + str(numberBlocks-1)]), nb_stacks=1, dilations=dilations, padding=paddingConv, use_skip_connections=True, dropout_rate=0.0, return_sequences=True,
                activation=activationConv) )
        Modell.add(GlobalMaxPooling1D())
    numberHidLayerDense = hyperparameter["numberHidLayerDense"]
    if numberHidLayerDense== int(numberHidLayerDense):
        numberHidLayerDense =  int(numberHidLayerDense)
    for i in range(0, numberHidLayerDense):
        layerSizeDense = hyperparameter["layerSizeDense" + str(i)]
        dropoutRateDense = hyperparameter["dropoutRateDense" + str(i)]
        Modell.add(Dense(layerSizeDense, activation=activationDense))
        Modell.add(Dropout(dropoutRateDense))
    if dataTask == "Regression":
        Modell.add(Dense(OutputLayer, activation='linear'))
    elif dataTask == "Classification":
        Modell.add(Dense(OutputLayer, activation='softmax'))
    return Modell

def DANN_MLP(InputLayer, OutputLayer, hyperparameter, activationDense, dataTask):
    
    L = {}
    L["featureExtractorInput"] = Input(shape=(InputLayer,), name="Feature_Extractor_Input_Layer")
    numberHidLayerDenseFeature = hyperparameter["numberHidLayerDenseFeature"]
    if numberHidLayerDenseFeature== int(numberHidLayerDenseFeature):
        numberHidLayerDenseFeature =  int(numberHidLayerDenseFeature)
    curLayer = 1
    L["L1"] = Dense(hyperparameter["layerSizeDenseFeature0"], activation=activationDense, name="Feature_Extractor_1")(L["featureExtractorInput"])
    curLayer = curLayer + 1
    if numberHidLayerDenseFeature > 1:
        L["L2"] = Dropout(hyperparameter["dropoutRateDenseFeature0"], name='Dropout_Feature_1')(L["L1"])
        curLayer = curLayer + 1
        curMLPFeature = 1
        for i in range(1, numberHidLayerDenseFeature-1):
            layerSizeDenseFeature = hyperparameter["layerSizeDenseFeature" + str(i)]
            dropoutRateDenseFeature = hyperparameter["dropoutRateDenseFeature" + str(i)]
            L["L"+str(curLayer)] = Dense(layerSizeDenseFeature, activation=activationDense, name='MLP_Feature_'+str(curMLPFeature))(L["L"+str(curLayer-1)])
            curLayer = curLayer + 1
            curMLPFeature = curMLPFeature + 1
            L["L"+str(curLayer)] = Dropout(dropoutRateDenseFeature, name='Dropout_Feature_'+str(curMLPFeature))(L["L"+str(curLayer-1)])
            curLayer = curLayer + 1
        L["L"+str(curLayer)] = Dense(hyperparameter['layerSizeDenseFeature' + str(numberHidLayerDenseFeature-1)], activation=activationDense, name='MLP_Feature_'+str(curMLPFeature))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
    L["featureExtractorOutput"] = Dropout(hyperparameter["dropoutRateDenseFeature" + str(numberHidLayerDenseFeature-1)], name='Feature_Output')(L["L"+str(curLayer-1)])
    L["L"+str(curLayer)] = Dense(hyperparameter['layerSizeDenseLabel0'], activation=activationDense, name='Input_Label_Path')(L["featureExtractorOutput"])
    curLayer = curLayer + 1
    L["L"+str(curLayer)] = Dropout(hyperparameter['dropoutRateDenseLabel0'], name='Dropout_Label_1')(L["L"+str(curLayer-1)])
    curLayer = curLayer + 1
    numberHidLayerDenseLabel = hyperparameter["numberHidLayerDenseLabel"]
    if numberHidLayerDenseLabel== int(numberHidLayerDenseLabel):
        numberHidLayerDenseLabel =  int(numberHidLayerDenseLabel)
    for i in range(1, numberHidLayerDenseLabel):
        layerSizeDenseLabel = hyperparameter["layerSizeDenseLabel" + str(i)]
        dropoutRateDenseLabel = hyperparameter["dropoutRateDenseLabel" + str(i)]
        L["L"+str(curLayer)] = Dense(layerSizeDenseLabel, activation=activationDense, name='MLP_Label_'+str(i))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
        L["L"+str(curLayer)] = Dropout(dropoutRateDenseLabel, name='Dropout_Label_'+str(i+1))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
    if dataTask == "Regression":
        L["labelPredictorOutput"] = Dense(OutputLayer, activation='linear', name='Label_Output')(L["L"+str(curLayer-1)])
    elif dataTask == "Classification":
        L["labelPredictorOutput"] = Dense(OutputLayer, activation="softmax", name='Label_Output')(L["L"+str(curLayer-1)])
    L["L"+str(curLayer)] = Dense(hyperparameter['layerSizeDenseDomain0'], activation=activationDense, name='Input_Domain_Path')(L["featureExtractorOutput"])
    curLayer = curLayer + 1
    L["L"+str(curLayer)] =  Dropout(hyperparameter['dropoutRateDenseDomain0'], name='Dropout_Domain_1')(L["L"+str(curLayer-1)])
    curLayer = curLayer + 1
    numberHidLayerDenseDomain = hyperparameter["numberHidLayerDenseDomain"]
    if numberHidLayerDenseDomain== int(numberHidLayerDenseDomain):
        numberHidLayerDenseDomain =  int(numberHidLayerDenseDomain)
    for i in range(1, numberHidLayerDenseDomain):
        layerSizeDenseDomain = hyperparameter["layerSizeDenseDomain" + str(i)]
        dropoutRateDenseDomain = hyperparameter["dropoutRateDenseDomain" + str(i)]
        L["L"+str(curLayer)] = Dense(layerSizeDenseDomain, activation=activationDense, name='MLP_Domain_'+str(i))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
        L["L"+str(curLayer)] = Dropout(dropoutRateDenseDomain, name='Dropout_Domain_'+str(i+1))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
    L["domainPredictorOutput"] = Dense(1, activation='sigmoid', name='Domain_Output')(L["L"+str(curLayer-1)])
    DANN_Modell = tf.keras.Model(inputs=L["featureExtractorInput"], outputs=[L["labelPredictorOutput"], L["domainPredictorOutput"]])
    return DANN_Modell

def DANN_TCN(InputLayer, OutputLayer, hyperparameter, activationDense, activationConv, paddingConv, paddingPool, dataTask):
    
    dilations = []
    for i in range(0,hyperparameter["numberLayerConvInBlock"]):
        dilations.append(2**i)
    L = {}
    L["featureExtractorInput"] = Input(batch_size=None, shape=(None,InputLayer), name="Feature_Extractor_Input_Layer")
    numberBlocks = hyperparameter["numberBlocks"]
    if numberBlocks== int(numberBlocks):
        numberBlocks =  int(numberBlocks)
    curLayer = 1
    L["L1"] = TCN(nb_filters=hyperparameter["numFiltersConv0"], kernel_size=(hyperparameter["filterSizeConv0"]), nb_stacks=1, dilations=dilations, padding=paddingConv, use_skip_connections=True, dropout_rate=0.0, return_sequences=True,
           activation=activationConv, name="Feature_Extractor_1")(L["featureExtractorInput"])
    curLayer = curLayer + 1
    if numberBlocks > 1:
        L["L2"] = MaxPooling1D(pool_size=hyperparameter["poolSizeConv0"], strides=hyperparameter["poolSizeConv0"], padding=paddingPool, name="Pooling_1")(L["L1"])
        curLayer = curLayer + 1
        curTCN = 1
        for i in range(1, numberBlocks-1):
            numFiltersConv = hyperparameter["numFiltersConv" + str(i)]
            filterSizeConv = hyperparameter["filterSizeConv" + str(i)]
            poolSizeConv = hyperparameter["poolSizeConv" + str(i)]
            L["L"+str(curLayer)] = TCN(nb_filters=numFiltersConv, kernel_size=(filterSizeConv), nb_stacks=1, dilations=dilations, padding=paddingConv, use_skip_connections=True, dropout_rate=0.0, return_sequences=True,
                   activation=activationConv, name='TCN_'+str(curTCN))(L["L"+str(curLayer-1)])
            curLayer = curLayer + 1
            curTCN = curTCN + 1
            L["L"+str(curLayer)] = MaxPooling1D(pool_size=poolSizeConv, strides=poolSizeConv, padding=paddingPool, name='Pooling_'+str(curTCN))(L["L"+str(curLayer-1)])
            curLayer = curLayer + 1
        L["L"+str(curLayer)] = TCN(nb_filters=hyperparameter["numFiltersConv" + str(numberBlocks-1)], kernel_size=(hyperparameter["filterSizeConv" + str(numberBlocks-1)]), nb_stacks=1, dilations=dilations, padding=paddingConv, use_skip_connections=True, dropout_rate=0.0, return_sequences=True,
                activation=activationConv, name='TCN_'+str(curTCN))(L["L"+str(curLayer-1)] )
        curLayer = curLayer + 1
    L["featureExtractorOutput"] = GlobalMaxPooling1D(name='Feature_Output')(L["L"+str(curLayer-1)])
    L["L"+str(curLayer)] = Dense(hyperparameter['layerSizeDenseLabel0'], activation=activationDense, name='Input_Label_Path')(L["featureExtractorOutput"])
    curLayer = curLayer + 1
    L["L"+str(curLayer)] = Dropout(hyperparameter['dropoutRateDenseLabel0'], name='Dropout_L1')(L["L"+str(curLayer-1)])
    curLayer = curLayer + 1
    numberHidLayerDenseLabel = hyperparameter["numberHidLayerDenseLabel"]
    if numberHidLayerDenseLabel== int(numberHidLayerDenseLabel):
        numberHidLayerDenseLabel =  int(numberHidLayerDenseLabel)
    for i in range(1, numberHidLayerDenseLabel):
        layerSizeDenseLabel = hyperparameter["layerSizeDenseLabel" + str(i)]
        dropoutRateDenseLabel = hyperparameter["dropoutRateDenseLabel" + str(i)]
        L["L"+str(curLayer)] = Dense(layerSizeDenseLabel, activation=activationDense, name='Label'+str(i))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
        L["L"+str(curLayer)] = Dropout(dropoutRateDenseLabel, name='Dropout_L'+str(i+1))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
    if dataTask == "Regression":
        L["labelPredictorOutput"] = Dense(OutputLayer, activation='linear', name='Label_Output')(L["L"+str(curLayer-1)])
    elif dataTask == "Classification":
        L["labelPredictorOutput"] = Dense(OutputLayer, activation="softmax", name='Label_Output')(L["L"+str(curLayer-1)])
    L["L"+str(curLayer)] = Dense(hyperparameter['layerSizeDenseDomain0'], activation=activationDense, name='Input_Domain_Path')(L["featureExtractorOutput"])
    curLayer = curLayer + 1
    L["L"+str(curLayer)] =  Dropout(hyperparameter['dropoutRateDenseDomain0'], name='Dropout_D1')(L["L"+str(curLayer-1)])
    curLayer = curLayer + 1
    numberHidLayerDenseDomain = hyperparameter["numberHidLayerDenseDomain"]
    if numberHidLayerDenseDomain== int(numberHidLayerDenseDomain):
        numberHidLayerDenseDomain =  int(numberHidLayerDenseDomain)
    for i in range(1, numberHidLayerDenseDomain):
        layerSizeDenseDomain = hyperparameter["layerSizeDenseDomain" + str(i)]
        dropoutRateDenseDomain = hyperparameter["dropoutRateDenseDomain" + str(i)]
        L["L"+str(curLayer)] = Dense(layerSizeDenseDomain, activation=activationDense, name='Domain'+str(i))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
        L["L"+str(curLayer)] = Dropout(dropoutRateDenseDomain, name='Dropout_D'+str(i+1))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
    L["domainPredictorOutput"] = Dense(1, activation='sigmoid', name='Domain_Output')(L["L"+str(curLayer-1)])
    DANN_Modell = tf.keras.Model(inputs=L["featureExtractorInput"], outputs=[L["labelPredictorOutput"], L["domainPredictorOutput"]])
    return DANN_Modell

def DANN_1DCNN(InputLayer, OutputLayer, hyperparameter, activationDense, activationConv, paddingConv, paddingPool, dataTask):
    
    L = {}
    L["featureExtractorInput"] = Input(batch_size=None, shape=(None,InputLayer), name="Feature_Extractor_Input_Layer")
    numberHidLayerConv = hyperparameter["numberHidLayerConv"]
    if numberHidLayerConv== int(numberHidLayerConv):
        numberHidLayerConv =  int(numberHidLayerConv)
    curLayer = 1
    L["L1"] = Conv1D(hyperparameter["numFiltersConv0"], kernel_size=(hyperparameter["filterSizeConv0"]), strides=1, padding=paddingConv, activation=activationConv, name="Feature_Extractor_1")(L["featureExtractorInput"])
    curLayer = curLayer + 1
    if numberHidLayerConv > 1:
        L["L2"] = MaxPooling1D(pool_size=hyperparameter["poolSizeConv0"], strides=hyperparameter["poolSizeConv0"], padding=paddingPool, name="Pooling_1")(L["L1"])
        curLayer = curLayer + 1
        curCNN = 1
        for i in range(1, numberHidLayerConv-1):
            numFiltersConv = hyperparameter["numFiltersConv" + str(i)]
            filterSizeConv = hyperparameter["filterSizeConv" + str(i)]
            poolSizeConv = hyperparameter["poolSizeConv" + str(i)]
            L["L"+str(curLayer)] = Conv1D(numFiltersConv, kernel_size=(filterSizeConv), strides=1, padding=paddingConv, activation=activationConv, name='CNN_'+str(curCNN))(L["L"+str(curLayer-1)])
            curLayer = curLayer + 1
            curCNN = curCNN + 1
            L["L"+str(curLayer)] = MaxPooling1D(pool_size=poolSizeConv, strides=poolSizeConv, padding=paddingPool, name='Pooling_'+str(curCNN))(L["L"+str(curLayer-1)])
            curLayer = curLayer + 1
        L["L"+str(curLayer)] = Conv1D(hyperparameter['numFiltersConv' + str(numberHidLayerConv-1)], kernel_size=(hyperparameter['filterSizeConv' + str(numberHidLayerConv-1)]), strides=1, padding=paddingConv, activation=activationConv, name='CNN_'+str(curCNN))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
    L["featureExtractorOutput"] = GlobalMaxPooling1D(name='Feature_Output')(L["L"+str(curLayer-1)])
    L["L"+str(curLayer)] = Dense(hyperparameter['layerSizeDenseLabel0'], activation=activationDense, name='Input_Label_Path')(L["featureExtractorOutput"])
    curLayer = curLayer + 1
    L["L"+str(curLayer)] = Dropout(hyperparameter['dropoutRateDenseLabel0'], name='Dropout_L1')(L["L"+str(curLayer-1)])
    curLayer = curLayer + 1
    numberHidLayerDenseLabel = hyperparameter["numberHidLayerDenseLabel"]
    if numberHidLayerDenseLabel== int(numberHidLayerDenseLabel):
        numberHidLayerDenseLabel =  int(numberHidLayerDenseLabel)
    for i in range(1, numberHidLayerDenseLabel):
        layerSizeDenseLabel = hyperparameter["layerSizeDenseLabel" + str(i)]
        dropoutRateDenseLabel = hyperparameter["dropoutRateDenseLabel" + str(i)]
        L["L"+str(curLayer)] = Dense(layerSizeDenseLabel, activation=activationDense, name='Label'+str(i))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
        L["L"+str(curLayer)] = Dropout(dropoutRateDenseLabel, name='Dropout_L'+str(i+1))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
    if dataTask == "Regression":
        L["labelPredictorOutput"] = Dense(OutputLayer, activation='linear', name='Label_Output')(L["L"+str(curLayer-1)])
    elif dataTask == "Classification":
        L["labelPredictorOutput"] = Dense(OutputLayer, activation="softmax", name='Label_Output')(L["L"+str(curLayer-1)])
    L["L"+str(curLayer)] = Dense(hyperparameter['layerSizeDenseDomain0'], activation=activationDense, name='Input_Domain_Path')(L["featureExtractorOutput"])
    curLayer = curLayer + 1
    L["L"+str(curLayer)] =  Dropout(hyperparameter['dropoutRateDenseDomain0'], name='Dropout_D1')(L["L"+str(curLayer-1)])
    curLayer = curLayer + 1
    numberHidLayerDenseDomain = hyperparameter["numberHidLayerDenseDomain"]
    if numberHidLayerDenseDomain== int(numberHidLayerDenseDomain):
        numberHidLayerDenseDomain =  int(numberHidLayerDenseDomain)
    for i in range(1, numberHidLayerDenseDomain):
        layerSizeDenseDomain = hyperparameter["layerSizeDenseDomain" + str(i)]
        dropoutRateDenseDomain = hyperparameter["dropoutRateDenseDomain" + str(i)]
        L["L"+str(curLayer)] = Dense(layerSizeDenseDomain, activation=activationDense, name='Domain'+str(i))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
        L["L"+str(curLayer)] = Dropout(dropoutRateDenseDomain, name='Dropout_D'+str(i+1))(L["L"+str(curLayer-1)])
        curLayer = curLayer + 1
    L["domainPredictorOutput"] = Dense(1, activation='sigmoid', name='Domain_Output')(L["L"+str(curLayer-1)])
    DANN_Modell = tf.keras.Model(inputs=L["featureExtractorInput"], outputs=[L["labelPredictorOutput"], L["domainPredictorOutput"]])
    return DANN_Modell

