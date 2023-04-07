# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:23:31 2023

@author: AriSpiesberger
"""
import subprocess
import importlib
import iterativeupdates
import numpy as np
def run_file_n_times(file_path, n):
    base_accuracy = []
    base_precision = []
    base_f1score = []
    base_recall = []
    semi_accuracy = []
    semi_precision = []
    semi_f1score = []
    semi_recall = []
    base_entropy = []
    semi_entropy = []
    
    for i in range(n):
        # Reload the iterativeupdates module for each iteration
        importlib.reload(iterativeupdates)
        
        # Capture the values of the desired variables
        base_accuracy.append(iterativeupdates.accuracy4)
        base_precision.append(iterativeupdates.precision4)
        base_recall.append(iterativeupdates.recall4)
        base_f1score.append(iterativeupdates.f1_4)
        semi_accuracy.append(iterativeupdates.accuracy44)
        semi_precision.append(iterativeupdates.precision44)
        semi_recall.append(iterativeupdates.recall44)
        semi_f1score.append(iterativeupdates.f1_44)
        base_entropy.append(iterativeupdates.base_ent)
        semi_entropy.append(iterativeupdates.semi_ent)
    
    return (base_accuracy, base_precision, base_f1score, base_recall,base_entropy,
            semi_accuracy, semi_precision, semi_f1score, semi_recall,semi_entropy)
        
        
a,b,c,d,e,f,g,h,j,k =   run_file_n_times("C:/Users.,/AriSpiesberger/Downloads/DavenAri/DavenAri/iterativeupdates.py", 100)