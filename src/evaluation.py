import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score,precision_score, recall_score, \
                            f1_score, confusion_matrix, ConfusionMatrixDisplay,\
                            precision_recall_curve


class Metrics:
    def __init__(self, y_hat, y) -> None:
        self.preds = y_hat
        self.labels = y
    
    def confusion_matrix(self, display = True, class_labels = None):
        matrix = confusion_matrix(self.labels, self.preds)

        if display == True:
            disp = ConfusionMatrixDisplay(matrix, 
                                        display_labels= class_labels)
            disp.plot(cmap = "autumn", colorbar = False)
            plt.show()
        else:
            return matrix
        
    def precision_recall_curve(self, model_type, threshold = .5):
        
        no_skill = len(self.labels[self.labels==1])/len(self.labels)
        precision, recall, thresholds = precision_recall_curve(self.labels,
                                                               self.preds)

        plt.plot(recall,precision, linestyle = "dashdot", label = model_type)
        plt.plot([0,1], [no_skill, no_skill], linestyle = "--", label = "No Skill")
        
        plt.xlabel("recall")
        plt.xlabel("precision")
        plt.legend()
        plt.show()
    
    def metrics(self):
        """
        Calculates accuracy, precision, recall, and f1 score for model predictions

        Returns dictionary containing these values
        """
        
        scores= {}
        scores['accuracy'] = accuracy_score(self.preds, self.labels)
        scores['precision'] = precision_score(self.preds, self.labels)
        scores['recall'] = recall_score(self.preds, self.labels)
        scores['f1'] = f1_score(self.preds, self.labels)
        

        print(scores)