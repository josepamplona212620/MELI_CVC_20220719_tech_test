from data_loader.meli_data_loader import MeliDataLoader, Meli_fashion_data_set
from models.conv_meli_fashion_model import ConvMeliModaModel
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
    # meli_data_set = Meli_fashion_data_set("training_data.csv")
    # meli_data_set.explore_dataset()
    # meli_data_set.print_data_categories()

    #  Create de data loader
    data_loader = MeliDataLoader(config)

    print('Create the model.')
    model = ConvMeliModaModel(config).model
    # model.model.summary()

    print('Create the trainer')
    trainer = ConvModelTrainer(model, data_loader.get_train_test_data(), config)

    for image, label in trainer.train_data.batch(2).take(1):
        print(label.numpy(), label.numpy().shape, image.numpy().shape)
        test_im = image.numpy()

    print(model.predict(test_im))
    # print('Start training the model.')
    # trainer.train()


if __name__ == '__main__':
    main()
