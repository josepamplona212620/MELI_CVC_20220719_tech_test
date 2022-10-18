from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

class ConvModelEvaluator:
    def __init__(self, model, data, checkpoint_path, batch):
        self.model = model.load_weigths(checkpoint_path)
        self.x = data[0]
        self.y = data[1]
        self.batch = batch
        self.predictions = []
        self.conf_mat = []
        self.accuracy = []
        self.f1 = []
        self.sensitivity = []
        self.specificity = []
        self.precision = []
        self.get_metrics()

    def get_metrics(self):
        self.predictions = self.model.predict(self.x, batch_size=self.batch)
        self.conf_mat = confusion_matrix(self.y, self.predictions)
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
