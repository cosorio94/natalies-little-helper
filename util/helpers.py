import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from fastai.text.all import *

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


repo_path = "/home/jupyter/src/natalies-little-helper/"
data_path = repo_path + "data/"
model_path = repo_path + "models/"
ulmfit_path = model_path + "ULMFiT/"
final_path = (Path(os.getcwd())).parent.absolute() / 'models' / 'ULMFiT' / 'final'


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

def ulmfit_dataloader(train_data, label):
    dataloader = DataBlock(blocks=(TextBlock.from_df('text', vocab=lang_dls.vocab, seq_len=72), CategoryBlock),
                 get_x=ColReader('text'),
                 get_y=ColReader(label),
                 splitter=RandomSplitter(valid_pct=0.2))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = intent_dls.dataloaders(train_data, bs=64, device=device)
    return dataloader

def load_dataloader(fname):
    return torch.load(fname)
    

# def load_ulmfit_classifier(class_type='intent', fname=f'{ulmfit_path}full_lang_dls_clean.pkl', class_fname=None, metrics=[accuracy, Perplexity()], wd=0.1):
    
#     if not class_fname:
#         if class_type == 'intent':
#             class_fname = f'{ulmfit_path}intent_main'
#         elif class_type == 'intent_groups':
#             class_fname = f'{ulmfit_path}intent_groups_main'
#         else:
#             class_fname = f'{ulmfit_path}sentiment_main'

#     dls = load_dataloader(fname)
#     classifier = text_classifier_learner(dls, AWD_LSTM, metrics=metrics, wd=wd, drop_mult=0.5).to_fp16()
#     classifier.load(class_fname)
#     return classifier

def load_ulmfit_classifier(class_type='intent', model_name=None, path=final_path):
    
    if not model_name:
        if class_type == 'intent':
            model_name = 'intent_classifier.pkl'
        elif class_type == 'intent_groups':
            model_name = 'intent_master_classifier.pkl'
        else:
            model_name = 'sentiment_classifier.pkl'

    classifier = load_learner(path / model_name)
    return classifier

