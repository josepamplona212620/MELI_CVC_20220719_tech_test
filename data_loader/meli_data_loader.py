from base.base_data_loader import BaseDataLoader
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from data_loader import img_procesing as proc

class MeliDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(MeliDataLoader, self).__init__(config)
        (self.x), (self.y) = Meli_moda_data_set(config).load_data()
        self.splitter_iterator = self.get_strstified_split(config)

    def get_strstified_split(self, config):
        skf = StratifiedKFold(n_splits=config.data_loader.splits, shuffle=True, random_state=config.seed)
        return iter(skf.split(self.x, self.y))

    def next_kfold(self):
        return next(self.splitter_iterator)

    def get_train_test_data(self):
        train_index, test_index = self.next_kfold()
        url_train_dataset = tf.data.Dataset.from_tensor_slices((
                                                                self.x.iloc[train_index].values,
                                                                self.y.iloc[train_index].values
                                                               ))
        url_test_dataset = tf.data.Dataset.from_tensor_slices((
                                                                self.x.iloc[test_index].values,
                                                                self.y.iloc[test_index].values
                                                              ))
        image_train_dataset = url_train_dataset.map(proc.tf_get_threshold_image).prefetch(tf.data.AUTOTUNE)
        image_test_dataset = url_test_dataset.map(proc.tf_get_threshold_image).prefetch(tf.data.AUTOTUNE)
        return image_train_dataset, image_test_dataset


class Meli_moda_data_set():
    def __init__(self, config):
        self.pandas_dataset = pd.read_csv(config.data_loader.csv_file)
        self.get_images_url()

    def get_images_url(self):
        self.pandas_dataset['picture_url'] = self.pandas_dataset['picture_id'].apply(
                                             lambda x: 'https://http2.mlstatic.com/D_{}-F.jpg'.format(x))

    def load_data(self, include_third_class=False):
        corrected_dataframe = self.pandas_dataset[self.pandas_dataset['picture_id'] != '699445-MLA50554255969_072022']
        if include_third_class:
            x = self.pandas_dataset['picture_id'].apply(
                lambda x: 'https://http2.mlstatic.com/D_{}-F.jpg'.format(x))
            y = self.pandas_dataset.astype({'correct_background?': 'int'})['correct_background?']
        else:
            binary_dataframe = corrected_dataframe[corrected_dataframe['correct_background?'] != '?']
            x = binary_dataframe['picture_id'].apply(
                lambda x: 'https://http2.mlstatic.com/D_{}-F.jpg'.format(x))
            y = binary_dataframe.astype({'correct_background?': 'int'})['correct_background?']
        return x, y

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
            url_request = urlopen(record.picture_url)
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

