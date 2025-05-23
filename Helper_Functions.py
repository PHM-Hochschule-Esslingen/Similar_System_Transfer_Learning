"""
Author: Marcel Braig
Version: 1.0
Contact: marcel.braig@hs-esslingen.de
"""

import numpy as np


def combineDics(listOfDics):
    finalCombinedDic = {}
    for mydict in listOfDics:
        for mylist in mydict:
            if type(mydict[mylist]) == list:
                if mylist in finalCombinedDic:
                    finalCombinedDic[mylist] = finalCombinedDic[mylist] + (mydict[mylist])
                else:
                    finalCombinedDic[mylist] = mydict[mylist]
            elif isinstance(mydict[mylist], (int, float, complex)):
                if mylist in finalCombinedDic:
                    finalCombinedDic[mylist] = finalCombinedDic[mylist] + [mydict[mylist]]
                else:
                    finalCombinedDic[mylist] = [mydict[mylist]] 
    return finalCombinedDic

def calculateMean(mydict):
    meanDict = {}
    for mylist in mydict:
        meanDict[mylist] = np.mean(mydict[mylist])
    
    return meanDict
