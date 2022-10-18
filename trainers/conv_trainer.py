from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from base.base_trainer import BaseTrain
import matplotlib.pyplot as plt
import os


class ConvModelTrainer(BaseTrain):
    def __init__(self, model, data, config):
        super(ConvModelTrainer, self).__init__(model, data, config)
        self.callbacks = []
        self.history = {}
        self.init_callbacks()

    def init_callbacks(self):
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
        self.history = self.model.fit(
            self.train_data.batch(self.config.trainer.batch_size),
            epochs=self.config.trainer.num_epochs,
            verbose=self.config.trainer.verbose_training,
            validation_data=self.test_data.batch(self.config.trainer.batch_size).take(2),
            callbacks=self.callbacks,
        )

    def show_train_history(self):
        keys_list = list(self.history.keys())
        hist = [self.history[key] for key in keys_list]
        # summarize history for accuracy
        plt.plot(hist[0])
        plt.plot(hist[1])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(hist[2])
        plt.plot(hist[3])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
