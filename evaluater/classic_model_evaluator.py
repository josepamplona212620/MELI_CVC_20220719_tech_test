from base.base_evaluator import BaseEvaluator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import pickle


class ConvModelEvaluator(BaseEvaluator):
    def __init__(self, model, data, checkpoint_path):
        super(ConvModelEvaluator, self).__init__(model)
        self.model = pickle.load(open(checkpoint_path, 'rb'))
        self.x = data[0]
        self.y = data[1]
        self.get_conf_mat_metrics()

    def get_conf_mat_metrics(self):
        self.predictions = self.model.predict_proba(self.x)[:, 1]
        self.conf_mat = confusion_matrix(self.y, np.rint(self.predictions))
        sns.heatmap(self.conf_mat, annot=True)
        self.get_metrics()

