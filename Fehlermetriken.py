"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import numpy as np
from sklearn.metrics import classification_report


def Fehlerprognosemetriken(time, labels, predictedLabels, alpha):

    prognosisMetric = {}
    if labels.ndim > 1:
        labels = np.squeeze(labels)
    if predictedLabels.ndim > 1:
        predictedLabels = np.squeeze(predictedLabels)
    errors = np.subtract(labels, predictedLabels)
    
    prognosisMetric["MSE"] = np.square(errors).mean()
    prognosisMetric["RMSE"] = np.sqrt(prognosisMetric["MSE"])
    prognosisMetric["MAE"] = np.abs(errors).mean()
    t_jEoL = time[-1]
    Positive_alpha_bound = labels + alpha * t_jEoL
    Negative_alpha_bound = labels - alpha * t_jEoL
    n_PHj = 0
    In_Boundry_List = []
    for i in range(len(predictedLabels)):
        if (predictedLabels[i] < Positive_alpha_bound[i]) & (predictedLabels[i] > Negative_alpha_bound[i]):
            n_PHj += 1
            In_Boundry_List.append(True)
        else:
            In_Boundry_List.append(False)
    reverseIn_Boundry_List = np.array( In_Boundry_List[::-1] )
    if all(reverseIn_Boundry_List == True):
        t_ja = time[0]
    elif reverseIn_Boundry_List[0] == True:
        changed_values_mask = reverseIn_Boundry_List[:-1] != reverseIn_Boundry_List[1:]
        indices_of_changed_values = np.where(changed_values_mask)[0]
        t_ja = time[- (indices_of_changed_values[0]+1)]
    else:
        t_ja = t_jEoL
    prognosisMetric["relPH"] = (t_jEoL - t_ja)/t_jEoL
    prognosisMetric["PHrate"] = n_PHj/len(In_Boundry_List)  
    
    return prognosisMetric
    

def Klassifikationsfehlermetriken(labels, predictedLabels):
    
    diagnosisMetric = classification_report(y_true=labels, y_pred=predictedLabels, output_dict=True)
    
    return diagnosisMetric





