from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class BaseEvaluator(object):
    def __init__(self, model):
        self.model = model
        self.x = []
        self.y = []
        self.predictions = []
        self.conf_mat = np.empty([2, 2], dtype=int)
        self.accuracy = []
        self.f1 = []
        self.sensitivity = []
        self.specificity = []
        self.precision = []
        self.comparations = []

    def get_metrics(self):
        tn, fp, fn, tp = self.conf_mat.ravel()
        self.accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.sensitivity = tp / (tp + fn)
        self.specificity = tn / (tn + fp)
        self.precision = tp / (tp + fn)
        self.f1 = 2 * self.sensitivity * self.specificity / (self.sensitivity + self.specificity)

    def get_precision_sensitivity_curves(self):
        precision, recall, thresholds = precision_recall_curve(self.y, self.predictions)
        fpr, tpr, _ = roc_curve(self.y, self.predictions)
        print('AUC metric: ', auc(fpr, tpr) * 100, '%\n')
        plt.plot(precision, recall)
        plt.ylabel('recall')
        plt.xlabel('Precision')
        plt.title('Curva ROC')
        plt.show()
        print('\n\n\n')
        fig, ax1 = plt.subplots()
        ax1.set_title('Curvas de sensibilidad y precisión')
        ax2 = ax1.twinx()
        ax1.plot(thresholds, recall[:-1], 'g-')
        ax2.plot(thresholds, precision[:-1], 'b-')
        ax1.set_xlabel('threshold')
        ax1.set_ylabel('Sensitivity')
        ax2.set_ylabel('Precision')
        ax1.yaxis.label.set_color('g')
        ax2.yaxis.label.set_color('b')
        plt.show()

    def get_predict_distributions(self):
        self.results = pd.DataFrame(
            list(zip(self.y.astype(int), np.rint(self.predictions).astype(int), self.predictions)),
            columns=['label', 'prediction', 'proba'])
        self.results[
            'tp_f_tn'] = self.results.label + self.results.prediction  # 0:true negatives,  1:misses, 2:true positives
        self.results['fp_tf'] = self.results.label - self.results.prediction
        self.results['tp_fp_tf_tn'] = self.results.apply(
            lambda x: x['tp_f_tn'] * x['fp_tf'] if x['fp_tf'] != 0 else x['tp_f_tn'], axis=1)
        groups = self.results.groupby(['tp_fp_tf_tn'])['proba']
        fp, tn, fn, tp = [groups.get_group(x) for x in groups.groups]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.set_title('Distribución de las predicciónes')
        ax1.hist(tn, label='Verdaderos Negativos', alpha=.4, edgecolor='green')
        ax1.hist(tp, label='Verdaderos Positivos', alpha=.4, edgecolor='green')
        ax2.hist(fn, label='Falsos Negativos', alpha=.4, edgecolor='red', color='r')
        ax2.hist(fp, label='Falsos Positivos', alpha=.4, edgecolor='red', color='g')

        ax1.set_xlabel('Probabilidad predicha')
        ax1.set_ylabel('Aciertos')
        ax2.set_ylabel('Fallos')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax2.set_ylim(0, 60)
        ax1.yaxis.label.set_color('g')
        ax2.yaxis.label.set_color('r')
        plt.show()

    def get_moderation_thresh(self, thresh_l, thresh_h):
        moderate_index = (self.results['proba'] > thresh_l) & (self.results['proba'] < thresh_h)

        moderate = self.results[(self.results['proba'] > thresh_l) & (self.results['proba'] < thresh_h)]
        trust = self.results[(self.results['proba'] <= thresh_l) | (self.results['proba'] >= thresh_h)]

        trust = trust.value_counts(normalize=True)
        thresh_acc = 100 * (trust[0] + trust[1]) / len(trust.index)
        thresh_fn = 100 * trust[1] / len(trust.index)
        thresh_fn2 = 100 * trust[1] / len(self.results.index)

        print("Tomando un intervalo para moderar: ", thresh_l, ' a ', thresh_h)
        print("la proporcion de imágenes a moderar es del : ", end=' ')
        print((len(trust.index) / len(self.results.index) * 100), '%', end='\n')
        print('Y el accuracy del las demas predicciones sería de: ', thresh_acc)
        print('donde solo se presentan el ', thresh_fn, '% de falsos negativos')
        print('y en comparación con el dataset de prueba un ', thresh_fn, '%')

