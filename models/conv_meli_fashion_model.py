from base.base_model import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten

class ConvMeliFashionModel(BaseModel):
    def __init__(self, config):
        super(ConvMeliFashionModel, self).__init__(config)
        self.in_shape = config.data_loader.image_shape
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(self.config.model.conv[0], kernel_size=(3, 3), padding='same',
                              activation='relu', input_shape=(self.in_shape, self.in_shape, 3)))

        for i in range(1, len(self.config.model.conv)):
            self.model.add(MaxPooling2D(pool_size=(self.config.model.pooling, self.config.model.pooling),
                                        strides=self.config.model.pooling, padding='same'))
            self.model.add(Conv2D(self.config.model.conv[i], (3, 3), activation='relu', padding='same'))

        self.model.add(MaxPooling2D(pool_size=(self.config.model.pooling, self.config.model.pooling),
                                    strides=self.config.model.pooling, padding='same'))
        self.model.add(Flatten())
        for units, activ_fn in zip(self.config.model.NN, self.config.model.activations):
            self.model.add(Dropout(self.config.model.drop))
            self.model.add(Dense(units, activation=activ_fn))

        self.model.compile(
              loss='binary_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])
