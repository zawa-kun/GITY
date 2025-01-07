# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:43:38 2024

@author: ryo
"""

import CalcHist
import numpy as np
from PIL import Image
import glob
import os

def Unknown_Hist(path):



    data=glob.glob(path + '*.jpg')
    X1=[]
    num_data = 0
    for i in data :
        X = np.array(Image.open(i))
        X1.append(X)
        num_data = num_data + 1

    will_delete = [False] * len(data)

    print(will_delete)


    for i in range(len(data)):
        if (i+1 >= num_data):
            break
        if (will_delete[i] == False):
            for j in range(len(data)-i-1):
                ans = CalcHist.CalcHist(X1[i], X1[i+j+1])
                if (ans > 0.85):
                    will_delete[i+j+1] = True;
                
        print("i increment")
    print(will_delete)
    
    for i in range(len(will_delete)):
        if will_delete[i] == True:
            print(data[i] + "is delete")
            os.remove(data[i])
    

    
