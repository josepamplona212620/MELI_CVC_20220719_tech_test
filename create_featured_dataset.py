from data_loader.feature_extractors import get_features_record, get_features_names
from data_loader.meli_data_loader import Meli_fashion_data_set
from utils.config import process_config
import csv

config = process_config("configs/meli_fashion_config.json")
feat_dataset = Meli_fashion_data_set(config)
threshold = 200
#
with open('features_dataset.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(get_features_names())
    for i in range(feat_dataset.size):
        image, label = feat_dataset.get_one_image_and_label(i)
        features_record = [label]
        features_record = features_record + get_features_record(image, threshold)
        writer.writerow(features_record)
        if i%37 == 0:
            print(i/37)

print('The dataset made of numeric features was stored as: "features_dataset.csv"')
