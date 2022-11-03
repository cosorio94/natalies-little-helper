import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_score(yTest, yPred):
    accuracy = accuracy_score(yTest, yPred)
    precisions = precision_score(yTest, yPred, average=None)
    recalls = recall_score(yTest, yPred, average=None)
    f1s = f1_score(yTest, yPred, average=None)

    print("Accuracy:\t", accuracy)
    print("Precision:\t", precisions)
    print("Recall: \t", recalls)
    print("F1 scores:\t", f1s)

    print(f"Average\n\tPrecision: {precision_score(yTest, yPred, average='weighted')}", end='\n\t')
    print(f"Recall: {recall_score(yTest, yPred, average='weighted')}", end='\n\t')
    print(f"F1: {f1_score(yTest, yPred, average='weighted')}")



