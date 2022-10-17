from data_loader.feature_extractors import get_features_record, get_features_names
from data_loader.meli_data_loader import Meli_fashion_data_set
import csv

feat_dataset = Meli_fashion_data_set("training_data.csv")
threshold = 200
save_batches = 37  # For memory cleaning

with open('features_dataset.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(get_features_names())

for j in range(int(feat_dataset.size/save_batches)):
    with open('features_dataset.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(save_batches):
            image, label = feat_dataset.get_one_image_and_label(i+j*save_batches)
            features_record = [label]
            features_record = features_record + get_features_record(image, threshold)
            writer.writerow(features_record)

    print('Saved batch #:', j+1, 'last record: ', i+j*save_batches)

print('The dataset made of numeric features was stored as: "features_dataset.csv"')
