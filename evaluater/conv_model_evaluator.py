from base.base_evaluator import BaseEvaluator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


class ConvModelEvaluator(BaseEvaluator):
    def __init__(self, model, data, checkpoint_path, batch, iterations=False):
        super(ConvModelEvaluator, self).__init__(model)
        self.model = model.load(checkpoint_path)
        self.x = data[0]
        self.y = data[1]
        self.batch = batch
        self.num_iter = iterations
        self.get_metrics()

    def get_conf_mat_metrics(self):
        if self.num_iter:
            self.predictions = self.model.predict(self.x.batch(self.batch).take(self.num_iter))[:, 1]
        else:
            self.predictions = self.model.predict(self.x.batch(self.batch))[:, 1]
        self.conf_mat = confusion_matrix(self.y, np.rint(self.predictions))
        sns.heatmap(self.conf_mat, annot=True)
        self.get_metrics()
