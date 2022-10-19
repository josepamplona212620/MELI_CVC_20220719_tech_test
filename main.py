from data_loader.meli_data_loader import MeliDataLoader, MeliFashionDataSet
from models.conv_meli_fashion_model import ConvMeliFashionModel
from evaluater.conv_model_evaluator import ConvModelEvaluator
from evaluater.classic_model_evaluator import C
from trainers.conv_trainer import ConvModelTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
import matplotlib.pyplot as plt

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    # Exploring the dataset
    meli_data_set = MeliFashionDataSet("training_data.csv")
    meli_data_set.explore_dataset()
    meli_data_set.print_data_categories()

    #  Create de data loader
    data_loader = MeliDataLoader(config)

    # 'Create the model.'
    model = ConvMeliFashionModel(config).model
    # model.summary()

    # Create the trainer
    trainer = ConvModelTrainer(model, data_loader.get_train_test_data(), config)
    print('Start training the model.')
    trainer.train()


    # Evaluation section
    model_weights_path = "experiments/2022-10-17/Meli_moda/checkpoints/Meli_moda-20-0.15.hdf5"
    config.dataloader.csv_file = "productive_data.csv"
    data_loader = MeliDataLoader(config)
    model = ConvMeliFashionModel(config).model
    evaluator = ConvModelEvaluator(model, data_loader.get_validation_data() , model_weights_path, 16)

if __name__ == '__main__':
    main()
