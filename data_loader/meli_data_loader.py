from base.base_data_loader import BaseDataLoader
import cv2
import urllib.request
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class MeliDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(MeliDataLoader, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = Meli_moda_data_set.load_data()
        self.X_train = self.X_train.reshape((-1, 28, 28, 1))
        self.X_test = self.X_test.reshape((-1, 28, 28, 1))

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

class Meli_moda_data_set():
    def __init__(self, config):
        self.pandas_dataset = pd.read_csv(config.data_loader.csv_file)
        self.get_images_url()

    def get_images_url(self):
        self.pandas_dataset['picture_url'] = self.pandas_dataset['picture_id'].apply(
                                             lambda x: 'https://http2.mlstatic.com/D_{}-F.jpg'.format(x))

    def print_data_categories(self):
        copyed_modit_dataset = self.pandas_dataset.copy()
        copyed_modit_dataset['domain_id'] = self.pandas_dataset['domain_id'].str.split('-').str[1]
        backgorund_types = copyed_modit_dataset['correct_background?'].value_counts()
        site_ids = copyed_modit_dataset['site_id'].value_counts()
        product_types = copyed_modit_dataset['domain_id'].value_counts()
        print('# of backgorund_types: ', str(len(backgorund_types)))
        print(backgorund_types.to_string(), '\n')
        print('# of site_ids: ', str(len(site_ids)))
        print(site_ids.to_string(), '\n')
        print('# of product_types: ', str(len(product_types)))
        print(product_types.to_string(), '\n')

    def explore_images_labels(self, df, axes, col):
        images = []
        titles = []
        for index, record in df.iterrows():
            url_request = urllib.request.urlopen(record.picture_url)
            image_in_bytes = np.asarray(bytearray(url_request.read()), dtype=np.uint8)
            image = cv2.imdecode(image_in_bytes, -1)
            resized_image = cv2.resize(image, (256, 256))
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            # images.append(rgb_image)
            # titles.append('Product_type: ' + record.domain_id + ', Background: ' + record['correct_background?'])
            axes[index, col].set_title(record.domain_id + ', bkgrd: ' + record['correct_background?'])
            axes[index, col].imshow(rgb_image)
        # return titles, images
    
    def explore_dataset(self):
        non_labeled_bagrounds = self.pandas_dataset[self.pandas_dataset['correct_background?'] == '?']
        labeled_bagrounds_0 = self.pandas_dataset[self.pandas_dataset['correct_background?'] == '0']
        labeled_bagrounds_1 = self.pandas_dataset[self.pandas_dataset['correct_background?'] == '1']

        dataset_samples_n = non_labeled_bagrounds.sample(n=3).reset_index()
        dataset_samples_0 = labeled_bagrounds_0.sample(n=3).reset_index()
        dataset_samples_1 = labeled_bagrounds_1.sample(n=3).reset_index()
        
        figure, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=True, sharex=True)
        self.explore_images_labels(dataset_samples_1, axes, 0)
        self.explore_images_labels(dataset_samples_n, axes, 1)
        self.explore_images_labels(dataset_samples_0, axes, 2)
        plt.show()
