"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

from hyperopt import hp
from hyperopt.pyll.base import scope


def Hyp_Opt_Parameter_Space(Net_type, data):
    
    if "DANN" in Net_type:
        if "MLP" in Net_type:
            minHidLayerDenseFeature = 1
            maxHidLayerDenseFeature = 5
            minHidLayerDenseLabel = 1
            maxHidLayerDenseLabel = 5
            minHidLayerDenseDomain = 1
            maxHidLayerDenseDomain = 5
            space = {
                'learnRate': hp.uniform('learnRate', 1e-5, 0.01,),
                'numberHidLayerDenseFeature': scope.int(hp.quniform('numberHidLayerDenseFeature', minHidLayerDenseFeature, maxHidLayerDenseFeature, 1)),
                'numberHidLayerDenseLabel': scope.int(hp.quniform('numberHidLayerDenseLabel', minHidLayerDenseLabel, maxHidLayerDenseLabel, 1)),
                'numberHidLayerDenseDomain': scope.int(hp.quniform('numberHidLayerDenseDomain', minHidLayerDenseDomain, maxHidLayerDenseDomain, 1))
            }
            if data == "Filter":
                space['batchSize'] = scope.int(hp.quniform('batchSize', 1, 500, 1))
            elif data == "Bearing":
                space['batchSize'] = scope.int(hp.quniform('batchSize', 1, 600, 1))
            if data == "Filter":
                space['lambdaValue'] = hp.uniform('lambdaValue', 0, 1)
            elif data == "Bearing":
                space['lambdaValue'] = hp.uniform('lambdaValue', 0, 1)
            for i in range(maxHidLayerDenseFeature):
                space['dropoutRateDenseFeature' + str(i)] = hp.uniform('dropoutRateDenseFeature' + str(i), 0, 0.8)
                space['layerSizeDenseFeature' + str(i)] = scope.int(hp.quniform('layerSizeDenseFeature' + str(i), 1, 100, 1))
            for i in range(maxHidLayerDenseLabel):
                space['dropoutRateDenseLabel' + str(i)] = hp.uniform('dropoutRateDenseLabel' + str(i), 0, 0.8)
                space['layerSizeDenseLabel' + str(i)] = scope.int(hp.quniform('layerSizeDenseLabel' + str(i), 1, 100, 1))
            for i in range(maxHidLayerDenseDomain):
                space['dropoutRateDenseDomain' + str(i)] = hp.uniform('dropoutRateDenseDomain' + str(i), 0, 0.8)
                space['layerSizeDenseDomain' + str(i)] = scope.int(hp.quniform('layerSizeDenseDomain' + str(i), 1, 100, 1))
        if "CNN" in Net_type:
            minHidLayerConv = 1
            maxHidLayerConv = 5
            minHidLayerDenseLabel = 1
            maxHidLayerDenseLabel = 5
            minHidLayerDenseDomain = 1
            maxHidLayerDenseDomain = 5
            space = {
                'learnRate': hp.uniform('learnRate', 1e-5, 0.01,),
                'numberHidLayerConv': scope.int(hp.quniform('numberHidLayerConv', minHidLayerConv, maxHidLayerConv, 1)),
                'numberHidLayerDenseLabel': scope.int(hp.quniform('numberHidLayerDenseLabel', minHidLayerDenseLabel, maxHidLayerDenseLabel, 1)),
                'numberHidLayerDenseDomain': scope.int(hp.quniform('numberHidLayerDenseDomain', minHidLayerDenseDomain, maxHidLayerDenseDomain, 1))
            }
            if data == "Filter":
                space['lambdaValue'] = hp.uniform('lambdaValue', 0, 1)
            elif data == "Bearing":
                space['lambdaValue'] = hp.uniform('lambdaValue', 0, 1)
            for i in range(maxHidLayerConv):
                space['numFiltersConv' + str(i)] = scope.int(hp.quniform('numFiltersConv' + str(i), 1, 30, 1))
                if data == "Filter":
                    space['filterSizeConv' + str(i)] = scope.int(hp.quniform('filterSizeConv' + str(i), 1, 20, 1))
                elif data == "Bearing":
                    space['filterSizeConv' + str(i)] = scope.int(hp.quniform('filterSizeConv' + str(i), 1, 20, 1))
            for i in range(maxHidLayerConv-1):
                if data == "Filter":
                    space['poolSizeConv' + str(i)] = scope.int(hp.quniform('poolSizeConv' + str(i), 1, 5, 1))
                elif data == "Bearing":
                    space['poolSizeConv' + str(i)] = scope.int(hp.quniform('poolSizeConv' + str(i), 1, 5, 1))
            for i in range(maxHidLayerDenseLabel):
                space['dropoutRateDenseLabel' + str(i)] = hp.uniform('dropoutRateDenseLabel' + str(i), 0, 0.8)
                space['layerSizeDenseLabel' + str(i)] = scope.int(hp.quniform('layerSizeDenseLabel' + str(i), 1, 50, 1))
            for i in range(maxHidLayerDenseDomain):
                space['dropoutRateDenseDomain' + str(i)] = hp.uniform('dropoutRateDenseDomain' + str(i), 0, 0.8)
                space['layerSizeDenseDomain' + str(i)] = scope.int(hp.quniform('layerSizeDenseDomain' + str(i), 1, 50, 1))
        elif "TCN" in Net_type:
            minBlocks = 1
            maxBlocks = 5
            minLayerConvInBlock = 2
            maxLayerConvInBlock = 5
            minHidLayerDenseLabel = 1
            maxHidLayerDenseLabel = 5
            minHidLayerDenseDomain = 1
            maxHidLayerDenseDomain = 5
            space = {
                'learnRate': hp.uniform('learnRate', 1e-5, 0.01,),
                'numberBlocks': scope.int(hp.quniform('numberBlocks', minBlocks, maxBlocks, 1)),
                'numberLayerConvInBlock': scope.int(hp.quniform('numberLayerConvInBlock', minLayerConvInBlock, maxLayerConvInBlock, 1)),
                'numberHidLayerDenseLabel': scope.int(hp.quniform('numberHidLayerDenseLabel', minHidLayerDenseLabel, maxHidLayerDenseLabel, 1)),
                'numberHidLayerDenseDomain': scope.int(hp.quniform('numberHidLayerDenseDomain', minHidLayerDenseDomain, maxHidLayerDenseDomain, 1))
            }   
            if data == "Filter":
                space['lambdaValue'] = hp.uniform('lambdaValue', 0, 1)
            elif data == "Bearing":
                space['lambdaValue'] = hp.uniform('lambdaValue', 0, 1)
            for i in range(maxBlocks):
                space['numFiltersConv' + str(i)] = scope.int(hp.quniform('numFiltersConv' + str(i), 1, 30, 1))
                if data == "Filter":
                    space['filterSizeConv' + str(i)] = scope.int(hp.quniform('filterSizeConv' + str(i), 1, 20, 1))
                elif data == "Bearing":
                    space['filterSizeConv' + str(i)] = scope.int(hp.quniform('filterSizeConv' + str(i), 1, 20, 1))
            for i in range(maxBlocks-1):
                if data == "Filter":
                    space['poolSizeConv' + str(i)] = scope.int(hp.quniform('poolSizeConv' + str(i), 1, 5, 1))
                elif data == "Bearing":
                    space['poolSizeConv' + str(i)] = scope.int(hp.quniform('poolSizeConv' + str(i), 1, 5, 1))
            for i in range(maxHidLayerDenseLabel):
                space['dropoutRateDenseLabel' + str(i)] = hp.uniform('dropoutRateDenseLabel' + str(i), 0, 0.8)
                space['layerSizeDenseLabel' + str(i)] = scope.int(hp.quniform('layerSizeDenseLabel' + str(i), 1, 50, 1))
            for i in range(maxHidLayerDenseDomain):
                space['dropoutRateDenseDomain' + str(i)] = hp.uniform('dropoutRateDenseDomain' + str(i), 0, 0.8)
                space['layerSizeDenseDomain' + str(i)] = scope.int(hp.quniform('layerSizeDenseDomain' + str(i), 1, 50, 1))
    else:
        if 'MLP' in Net_type:
            minHidLayer = 1
            maxHidLayer = 10
            space = {
                'numberHidLayer': scope.int(hp.quniform('numberHidLayer', minHidLayer, maxHidLayer, 1)),
                'learnRate': hp.uniform('learnRate', 1e-5, 0.01),
                'learnRateFinetuning': hp.uniform('learnRateFinetuning', 1e-5, 0.01),
                'lastLayersRetrainPercentage': hp.uniform('lastLayersRetrainPercentage', 0, 1)
            }
            if data == "Filter":
                space['batchSize'] = scope.int(hp.quniform('batchSize', 1, 500, 1))
            elif data == "Bearing":
                space['batchSize'] = scope.int(hp.quniform('batchSize', 1, 600, 1))
            for i in range(maxHidLayer):
                space['dropoutRate' + str(i)] = hp.uniform('dropoutRate' + str(i), 0, 0.8)
                space['layerSize' + str(i)] = scope.int(hp.quniform('layerSize' + str(i), 1, 100, 1))
        elif 'CNN' in Net_type:
            minHidLayerConv = 1
            maxHidLayerConv = 5
            minHidLayerDense = 1
            maxHidLayerDense = 5
            space = {
                'numberHidLayerConv': scope.int(hp.quniform('numberHidLayerConv', minHidLayerConv, maxHidLayerConv, 1)),
                'numberHidLayerDense': scope.int(hp.quniform('numberHidLayerDense', minHidLayerDense, maxHidLayerDense, 1)),
                'learnRate': hp.uniform('learnRate', 1e-5, 0.01),
                'learnRateFinetuning': hp.uniform('learnRateFinetuning', 1e-5, 0.01),
                'lastLayersRetrainPercentage': hp.uniform('lastLayersRetrainPercentage', 0, 1)
            }    
            for i in range(maxHidLayerConv):
                space['numFiltersConv' + str(i)] = scope.int(hp.quniform('numFiltersConv' + str(i), 1, 30, 1))
                if data == "Filter":
                    space['filterSizeConv' + str(i)] = scope.int(hp.quniform('filterSizeConv' + str(i), 1, 20, 1))
                elif data == "Bearing":
                    space['filterSizeConv' + str(i)] = scope.int(hp.quniform('filterSizeConv' + str(i), 1, 20, 1))
            for i in range(maxHidLayerConv-1):
                if data == "Filter":
                    space['poolSizeConv' + str(i)] = scope.int(hp.quniform('poolSizeConv' + str(i), 1, 5, 1))
                elif data == "Bearing":
                    space['poolSizeConv' + str(i)] = scope.int(hp.quniform('poolSizeConv' + str(i), 1, 5, 1))
            for i in range(maxHidLayerDense):
                space['dropoutRateDense' + str(i)] = hp.uniform('dropoutRateDense' + str(i), 0, 0.8)
                space['layerSizeDense' + str(i)] = scope.int(hp.quniform('layerSizeDense' + str(i), 1, 50, 1))
        elif 'TCN' in Net_type: 
            minBlocks = 1
            maxBlocks = 5
            minLayerConvInBlock = 2
            maxLayerConvInBlock = 5
            minHidLayerDense = 1
            maxHidLayerDense = 5
            space = {
                'numberBlocks': scope.int(hp.quniform('numberBlocks', minBlocks, maxBlocks, 1)),
                'numberLayerConvInBlock': scope.int(hp.quniform('numberLayerConvInBlock', minLayerConvInBlock, maxLayerConvInBlock, 1)),
                'numberHidLayerDense': scope.int(hp.quniform('numberHidLayerDense', minHidLayerDense, maxHidLayerDense, 1)),
                'learnRate': hp.uniform('learnRate', 1e-5, 0.01),
                'learnRateFinetuning': hp.uniform('learnRateFinetuning', 1e-5, 0.01),
                'lastLayersRetrainPercentage': hp.uniform('lastLayersRetrainPercentage', 0, 1)
            }
            for i in range(maxBlocks):
                space['numFiltersConv' + str(i)] = scope.int(hp.quniform('numFiltersConv' + str(i), 1, 30, 1))
                if data == "Filter":  
                    space['filterSizeConv' + str(i)] = scope.int(hp.quniform('filterSizeConv' + str(i), 1, 20, 1))
                elif data == "Bearing": 
                    space['filterSizeConv' + str(i)] = scope.int(hp.quniform('filterSizeConv' + str(i), 1, 20, 1))
            for i in range(maxBlocks-1):
                if data == "Filter":
                    space['poolSizeConv' + str(i)] = scope.int(hp.quniform('poolSizeConv' + str(i), 1, 5, 1))
                elif data == "Bearing":
                    space['poolSizeConv' + str(i)] = scope.int(hp.quniform('poolSizeConv' + str(i), 1, 5, 1))
            for i in range(maxHidLayerDense):
                space['dropoutRateDense' + str(i)] = hp.uniform('dropoutRateDense' + str(i), 0, 0.8)
                space['layerSizeDense' + str(i)] = scope.int(hp.quniform('layerSizeDense' + str(i), 1, 50, 1))
    
    return space

