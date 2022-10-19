from data_loader.img_procesing import get_url_image, resize_image
from base.base_evaluator import BaseEvaluator
from sklearn.metrics import confusion_matrix
import matplotlib as plt
import seaborn as sns
import numpy as np


class ConvModelEvaluator(BaseEvaluator):
    def __init__(self, model, data, checkpoint_path, batch, urls):
        super(ConvModelEvaluator, self).__init__(model)
        self.model = model.load(checkpoint_path)
        self.x = data[0]
        self.y = data[1]
        self.image_urls = urls
        self.batch = batch
        self.get_metrics()

    def get_conf_mat_metrics(self):
        self.predictions = self.model.predict(self.x.batch(self.batch))[:, 1]
        self.conf_mat = confusion_matrix(self.y, np.rint(self.predictions))
        sns.heatmap(self.conf_mat, annot=True)
        self.get_metrics()

    def get_bad_results_with_image(self, num_examples):
        self.results['urls'] = self.image_urls
        false_positives = self.results[self.results['tp_fp_tf_tn'] == -1.0].reset_index()
        false_negatives = self.results[self.results['tp_fp_tf_tn'] == 1.0].reset_index()

        fig = plt.figure(figsize=(8, 4*num_examples))
        for i in range(1, 2* num_examples + 1, 2):
            img = resize_image(get_url_image(false_positives.iloc[i]['urls']))
            fig.add_subplot(num_examples, 2, i)
            plt.imshow(img)
            img = resize_image(get_url_image(false_negatives.iloc[i]['urls']))
            fig.add_subplot(num_examples, 2, i+1)
            plt.imshow(img)
        fig.axes[-2].set_xlabel('Falsos Positivos')
        fig.axes[-1].set_xlabel('Falsos Negativos')
        plt.show()

