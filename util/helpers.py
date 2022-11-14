import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


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


def evaluate_model(model, X_test, y_test, labels):
    '''
    Uses the passed model to make predictions on the passed test data, and compares
    it to the real labels to print out the model's precision, recall, and f1-score.
    Returns the predictions made by the model.
    '''
    y_pred = model.predict(X_test)
    
    # prints out the model evaluation metrics
    report = classification_report(y_test, y_pred)
    print(report)

    # Plot confustion matrix
    cmp = ConfusionMatrixDisplay(confusion_matrix(
        y_test, y_pred), display_labels=labels)#, display_labels=y_train_df.astype('category').cat.categories)
    cmp.plot()
    plt.show()

    return y_pred