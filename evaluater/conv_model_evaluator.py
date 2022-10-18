from base.base_evaluator import BaseEvaluator
from sklearn.metrics import confusion_matrix
import seaborn as sns

class ConvModelEvaluator(BaseEvaluator):
    def __init__(self, model, data, checkpoint_path, batch):
        super(ConvModelEvaluator, self).__init__(model, data)
        self.model = model.load_weigths(checkpoint_path)
        self.x = data[0]
        self.y = data[1]
        self.batch = batch
        self.get_metrics()

    def get_conf_mat_metrics(self):
        self.predictions = self.model.predict_proba(self.x, batch_size=self.batch)
        self.conf_mat = confusion_matrix(self.y, self.predictions)
        sns.heatmap(self.conf_mat, annot=True)
        self.get_metrics()
