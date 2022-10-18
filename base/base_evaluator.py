from sklearn.metrics import  precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

class BaseEvaluator(object):
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.predictions = []
        self.conf_mat = np.empty([2, 2], dtype=int)
        self.accuracy = []
        self.f1 = []
        self.sensitivity = []
        self.specificity = []
        self.precision = []

    def get_metrics(self):
        tn, fp, fn, tp = self.conf_mat.ravel()
        self.accuracy = (tp+tn)/(tp+tn+fp+fn)
        self.sensitivity = tp/(tp+fn)
        self.specificity = tn/(tn+fp)
        self.precision = tp/(tp+fn)
        self.f1 = 2*self.sensitivity*self.specificity/(self.sensitivity+self.specificity)

    def get_accuracy_sensitivity_curves(self):
        precision, recall, thresholds = precision_recall_curve(self.y, self.predictions)
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(thresholds, recall, 'g-')
        ax2.plot(thresholds, precision, 'b-')
        ax1.set_xlabel('threshold')
        ax1.set_ylabel('Sensitivity')
        ax2.set_ylabel('Precision')
        plt.show()
