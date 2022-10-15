from data_loader.meli_data_loader import MeliDataLoader, Meli_moda_data_set
# from models.simple_mnist_model import SimpleMnistModel
# from trainers.simple_mnist_trainer import SimpleMnistModelTrainer
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
    # meli_data_set = Meli_moda_data_set(config)
    # meli_data_set.explore_dataset()
    # meli_data_set.print_data_categories()

    #  Create de data loader
    data_loader = MeliDataLoader(config)
    image_train_dataset, image_test_dataset = data_loader.get_train_test_data()

    # print some examples of processed dataset
    # for image, label in image_test_dataset.shuffle(10).take(1):
    #     print(label.numpy(), label.numpy().shape, image.numpy().shape)
    #     plt.imshow(image.numpy())
    #     plt.show()


    # print('Create the model.')
    # model = SimpleMnistModel(config)
    #
    # print('Create the trainer')
    # trainer = SimpleMnistModelTrainer(model.model, data_loader.get_train_data(), config)
    #
    # print('Start training the model.')
    # trainer.train()


if __name__ == '__main__':
    main()
