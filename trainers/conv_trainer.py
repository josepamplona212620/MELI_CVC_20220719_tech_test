from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from base.base_trainer import BaseTrain
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


class ConvModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(ConvModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.history = {}
        self.init_callbacks()

    def init_callbacks(self):
        """
        instance callbacks for save and log model and training based on
        the configuration in the config.json file

        Returns
        -------
        None
        """
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                                      '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

    def train(self):
        """
        train the instance model based on the configuration in the config.json file

        Returns
        -------
        None
        """
        self.reset_seed()
        self.history = self.model.fit(
            self.train_data.batch(self.config.trainer.batch_size),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=self.test_data.batch(self.config.trainer.batch_size).take(2),
            callbacks=self.callbacks,
        )

    def show_train_history(self):
        """
        Plot the data stored during training

        Returns
        -------
        None
        """
        keys_list = list(self.history.history.keys())
        hist = [self.history.history[key] for key in keys_list]
        # summarize history for accuracy
        plt.plot(hist[0])
        plt.plot(hist[2])
        plt.title(keys_list[0]+' vs '+keys_list[2])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(hist[1])
        plt.plot(hist[3])
        plt.title(keys_list[1]+' vs '+keys_list[3])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def reset_seed(self):
        np.random.RandomState(self.config.seed)
        tf.keras.utils.set_random_seed(self.config.seed)
