from base.base_data_loader import BaseDataLoader
from urllib.request import urlopen
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from data_loader import img_procesing as proc

class MeliDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(MeliDataLoader, self).__init__(config)
        (self.x), (self.y) = MeliFashionDataSet(config.data_loader.csv_file).get_urls_and_labels()
        self.splitter_iterator = self.get_stratified_split()

    def get_stratified_split(self):
        """
        Divides in stratified way the dataset into specified splits and creates an iterator to
        uses in a K-fold cross validation training
        Returns firs iteration of the k-fold object

        Returns
        -------
        Training and testing dataset from first Stratified K-Fold iteration
        """
        skf = StratifiedKFold(n_splits=self.config.data_loader.splits, shuffle=True, random_state=self.config.seed)
        return iter(skf.split(self.x, self.y))

    def next_kfold(self):
        """
        Iterates over a k-fold iterator to get next training-testing dataset
        Returns the next iteration of the k-fold object

        Returns
        -------
        Training and testing dataset from next Stratified K-Fold iteration
        """
        return next(self.splitter_iterator)

    def get_train_test_data(self):
        """
        Uses a k-fold iterator to get train and test dataset split to turnit into
        tf.dataset
        The tf.dataset object is used to apply the data input pipeline from url to normalized
        thresholded square rgb image
        Return training and testing tf.datasets with images and one hot encoded labels

        Returns
        -------
        Training and testing dataset from next Stratified K-Fold iteration
        """
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

    def get_validation_data(self):
        """
        Creates a tf.dataset from urls
        The tf.dataset object is used to apply the data input pipeline from url to normalized
        thresholded square rgb image

        Returns
        -------
        list
            - validation images (tf.dataset)
            - integer labels
        """
        url_valid_dataset = tf.data.Dataset.from_tensor_slices(self.x.values)
        image_valid_dataset = url_valid_dataset.map(proc.tf_get_thresh_valid_image).prefetch(tf.data.AUTOTUNE)
        return [image_valid_dataset, self.y]


class MeliFashionDataSet(object):
    def __init__(self, csv_file):
        self.pandas_dataset = pd.read_csv(csv_file)
        self.corrected_dataframe = []
        self.x = []
        self.y = []
        self.get_urls_and_labels()
        self.size = len(self.corrected_dataframe.index)

    def get_urls_and_labels(self, include_third_class=False):
        """
        Get urls and labels from pandas dataset
        There is a third not specific class that is dropped based on include_third_class flag

        Parameters
        ----------
        include_third_class: bool
            flag to drop not specific class

        Returns
        -------
        x: pandas.Series
            urls where images are stored
        y: padnas.Series
            integer labels
        """
        self.corrected_dataframe = self.pandas_dataset[
            self.pandas_dataset['picture_id'] != '699445-MLA50554255969_072022'].reset_index()  # it's corrupted data
        if not include_third_class:
            self.corrected_dataframe = self.corrected_dataframe[
                self.corrected_dataframe['correct_background?'] != '?'].reset_index()
            self.y = self.corrected_dataframe.astype({'correct_background?': 'int'})['correct_background?']
        else:
            self.y = self.corrected_dataframe['correct_background?']
        self.x = self.corrected_dataframe['picture_id'].apply(
            lambda l: 'https://http2.mlstatic.com/D_{}-F.jpg'.format(l))
        return self.x, self.y

    def print_data_categories(self):
        """
        Print the data categories of 3 columns in the pandas dataset

        Returns
        -------
        None
        """
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
        """
        Plot example images on a subplots grid columns

        Parameters
        ----------
        df: pandas.DataFrame
            sampled dataframe with specific label
        axes: pyplot.axes
            axes where images will be plotted
        col: int
            exes column variable where the images will be plotted
        """
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

    def explore_dataset(self, seed=None):
        """
        Plot random examples of the images in the dataset on a subplots grid
        showing the differences between the 3 classes

        Parameters
        ----------
        seed: int
            value to be able to reproduce the random sampling of the images to show
        """
        self.get_urls_and_labels(include_third_class=True)
        self.corrected_dataframe['picture_url'] = self.x
        non_labeled_bagrounds = self.corrected_dataframe[self.corrected_dataframe['correct_background?'] == '?']
        labeled_bagrounds_0 = self.corrected_dataframe[self.corrected_dataframe['correct_background?'] == '0']
        labeled_bagrounds_1 = self.corrected_dataframe[self.corrected_dataframe['correct_background?'] == '1']

        dataset_samples_n = non_labeled_bagrounds.sample(n=3, random_state=seed).reset_index()
        dataset_samples_0 = labeled_bagrounds_0.sample(n=3, random_state=seed).reset_index()
        dataset_samples_1 = labeled_bagrounds_1.sample(n=3, random_state=seed).reset_index()

        figure, axes = plt.subplots(3, 3, figsize=(15, 15), sharey=True, sharex=True)
        self.explore_images_labels(dataset_samples_1, axes, 0)
        self.explore_images_labels(dataset_samples_n, axes, 1)
        self.explore_images_labels(dataset_samples_0, axes, 2)
        plt.show()

    def get_image_from_url(self, picture_url):
        """
        Get RGB image from an url stored in a str variable

        Parameters
        ----------
        picture_url: str
            URL where an image is stored

        Returns
        -------
        rgb_img: numpy.ndarray
            rgb image
        """
        with urlopen(picture_url) as request:
            image_file_in_bytes = np.asarray(bytearray(request.read()), dtype=np.uint8)
        decoded_img = cv2.imdecode(image_file_in_bytes, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
        return rgb_img

    def get_one_image_and_label(self, record_index):
        """
        Get RGB image and its label from a pandas record with url and label

        Parameters
        ----------
        record_index: pandas.Series
            url and label of an example in the dataset

        Returns
        -------
        image: numpy.ndarray
            rgb image sample
        label: int
            integer sample label
        """
        image = self.get_image_from_url(self.x[record_index])
        label = self.y[record_index]
        return image, label