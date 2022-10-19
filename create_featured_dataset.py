from data_loader.feature_extractors import get_features_record, get_features_names
from data_loader.meli_data_loader import MeliFashionDataSet
from pathlib import Path
import csv


feat_dataset = MeliFashionDataSet("productive_data.csv")
threshold = 200
save_batches = 53  # For saving in intervals
file_name = "features_validation_dataset.csv"

if Path(file_name).is_file():
    print('This dataset exist: The feature records will be appended to it')
else:
    print('Csv file, Created.')
    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(get_features_names())

print("it`ll take ", feat_dataset.size/save_batches, ' batches')
for j in range(int(feat_dataset.size/save_batches)):
    with open(file_name, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(save_batches):
            image, label = feat_dataset.get_one_image_and_label(i+j*save_batches)
            features_record = [label]
            features_record = features_record + get_features_record(image, threshold)
            writer.writerow(features_record)

    print('Saved batch #:', j+1, 'last record: ', i+j*save_batches)

print('The dataset made of numeric features was stored as: "features_dataset.csv"')
