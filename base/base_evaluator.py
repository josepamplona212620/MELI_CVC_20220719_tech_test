from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class BaseEvaluator(object):
    def __init__(self, model):
        self.model = model
        self.x = np.empty([1], dtype=int)
        self.y = np.empty([1], dtype=int)
        self.predictions = []
        self.conf_mat = np.empty([2, 2], dtype=int)
        self.accuracy = []
        self.f1 = []
        self.sensitivity = []
        self.specificity = []
        self.precision = []
        self.results = pd.DataFrame()
        self.result_dictionary = {
            -1.0: 'Falsos Positivos',
            0.0: 'Verdaderos Negativos',
            1.0: 'Falsos Negativos',
            2.0: 'Verdaderos Positivos'
        }


    def get_metrics(self):
        """
        Calculates the confusion matrix and 5 metrics from this one.
        Also create a results Dataframe with all the results in the evaluation

        Returns
        -------
        None
        """
        tn, fp, fn, tp = self.conf_mat.ravel()
        self.accuracy = (tp + tn) / (tp + tn + fp + fn)
        self.sensitivity = tp / (tp + fn)
        self.specificity = tn / (tn + fp)
        self.precision = tp / (tp + fn)
        self.f1 = 2 * self.sensitivity * self.specificity / (self.sensitivity + self.specificity)
        self.get_results_dataset()

    def get_results_dataset(self):
        """
        Create results Dataframe by zipping the labels, predictions and probability predictions and label them
        with the result_dictionary of te class.

        Returns
        -------
        None
        """
        self.results = pd.DataFrame(
            list(zip(self.y.astype(int), np.rint(self.predictions).astype(int), self.predictions)),
            columns=['label', 'prediction', 'proba'])
        self.results[
            'tp_f_tn'] = self.results.label + self.results.prediction  # 0:true negatives,  1:misses, 2:true positives
        self.results['fp_tf'] = self.results.label - self.results.prediction
        self.results['tp_fp_tf_tn'] = self.results.apply(
            lambda x: x['tp_f_tn'] * x['fp_tf'] if x['fp_tf'] != 0 else x['tp_f_tn'], axis=1)

    def get_ROC_curves(self):
        """
        Print the AUC metric and plot the ROC curve as a binary classification metric

        Returns
        -------
        None
        """
        fpr, tpr, _ = roc_curve(self.y, self.predictions)
        print('AUC metric: ', auc(fpr, tpr) * 100, '%\n')

        plt.plot(fpr, tpr)
        plt.ylabel('recall')
        plt.xlabel('Precision')
        plt.title('Curva ROC')
        plt.show()

    def get_precision_sensitivity_curves(self):
        """
        Plot the Sensitivity and precision curves with respect to the change in threshold election.

        Returns
        -------
        None
        """
        precision, recall, thresholds = precision_recall_curve(self.y, self.predictions)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(thresholds, recall[:-1], 'g-')
        ax2.plot(thresholds, precision[:-1], 'b-')

        ax1.set_xlabel('threshold')
        ax1.set_ylabel('Sensitivity')
        ax2.set_ylabel('Precision')
        ax1.set_title('Curvas de sensibilidad y precisión')
        ax1.yaxis.label.set_color('g')
        ax2.yaxis.label.set_color('b')
        plt.show()

    def get_predict_distributions(self):
        """
        Plot a distribution of 'True Positives', 'True Negatives', 'False Positives' and 'False Negatives'
        in order to identify which thresholds can segment them

        Returns
        -------
        None
        """
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
        """
        Print the effects on excluding some results that can be analyzed later

        Returns
        -------
        None
        """
        moderate = self.results[(self.results['proba'] > thresh_l) & (self.results['proba'] < thresh_h)]
        trust = self.results[(self.results['proba'] <= thresh_l) | (self.results['proba'] >= thresh_h)]

        trust_counts = trust['tp_fp_tf_tn'].value_counts(normalize=True).sort_index()
        thresh_acc = 100*(trust_counts[0.0]+trust_counts[2.0])
        thresh_fn = 100*trust_counts[1.0]
        thresh_fn2 = 100*len(trust.index)*trust_counts[1.0]/len(self.results.index)

        print("Tomando un intervalo para moderar: ", thresh_l,' a ', thresh_h)
        print("la proporcion de imágenes a moderar es del : ", end=' ')
        print((100*len(moderate.index)/len(self.results.index)),'%', end='\n')
        print('Y el accuracy del las demas predicciones sería de: ', thresh_acc)
        print('donde solo se presentan el ', thresh_fn, '% de falsos negativos')
        print('y en comparación con el dataset de prueba un ', thresh_fn2, '%')
