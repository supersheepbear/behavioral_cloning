import json
import preprocess
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


class BaseModel(preprocess.ProcessData):
    """Base training model class
    This class contains all the training layers and training process
    """
    def __init__(self):
        """Initialization of training model
        In this model, the only hyper parameter to tune is batch size,
        which is assigned here.
        """
        preprocess.ProcessData.__init__(self)
        self.batch_size = 512

    def train_model(self):
        """main training model
        """
        model = Sequential()
        model.add(Lambda(lambda x: (x - 127.5) / 127.5, input_shape=(60, 320, 3)))
        model.add(Conv2D(24, 5, 5, name='conv1', subsample=(2, 2)))
        model.add(ELU())
        model.add(Conv2D(36, 5, 5, name='conv2', subsample=(2, 2)))
        model.add(ELU())
        model.add(Conv2D(48, 5, 5, name='conv3', subsample=(2, 2)))
        model.add(ELU())
        model.add(Conv2D(64, 3, 3, name='conv4', subsample=(1, 1)))
        model.add(ELU())
        model.add(Conv2D(64, 2, 2, name='conv5', subsample=(1, 1)))
        model.add(ELU())
        model.add(Flatten(name='fc0'))
        model.add(Dense(100, name='fc1'))
        model.add(ELU())
        model.add(Dense(50, name='fc2'))
        model.add(ELU())
        model.add(Dense(10, name='fc3'))
        model.add(ELU())
        model.add(Dense(1, name='output'))
        model.summary()
        model.compile(loss='mse', optimizer='Adam')
        # Set an early stop call back to make model stop training if validation loss increases
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0001,
                                   patience=0,
                                   verbose=0,
                                   mode='auto')

        # Set a callback to save model in each epoch
        check_pointer = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=False,
                                        mode='auto',
                                        period=1)

        # Save model in a history object for later plotting
        history_object = model.fit_generator(self.train_generator,
                                             samples_per_epoch=self.train_log.shape[0],
                                             validation_data=self.validation_generator,
                                             nb_val_samples=self.valid_log.shape[0],
                                             nb_epoch=15,
                                             callbacks=[check_pointer, early_stop],
                                             verbose=1)

        # Save model to h5 and json formats
        model.save('final_model.h5')
        with open('model.json', 'w') as f:
            json.dump(model.to_json(), f)

        # Plot training loss and validation loss
        self.plotting(history_object)

        # Evaluate test loss
        test_x, test_y = self.prepare_test_data(self.test_log)
        print(np.array(test_x).shape, np.array(test_y).shape)
        evaluation = model.evaluate(np.array(test_x), np.array(test_y))
        print('test loss: {}'.format(evaluation))

    @staticmethod
    def plotting(history_object):
        """Function for plotting from training/validation loss from history object
        """
        print(history_object.history.keys())
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()


def main():
    """Main function of the model
    """
    my_model = BaseModel()
    my_model.read_csv_data()
    my_model.create_generator()
    my_model.train_model()


if __name__ == "__main__":
    main()
