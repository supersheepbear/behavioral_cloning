import json
import preprocess
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


class BaseModel(preprocess.ProcessData):

    def __init__(self):
        preprocess.ProcessData.__init__(self)
        self.batch_size = 256

    def train_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: (x - 127.5) / 127.5, input_shape=(56, 128, 3)))
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
        model.add(Dense(1164, name='fc1'))
        model.add(ELU())
        model.add(Dense(100, name='fc2'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Dense(50, name='fc3'))
        model.add(ELU())
        model.add(Dropout(0.5))
        model.add(Dense(10, name='fc4'))
        model.add(ELU())
        model.add(Dense(1, name='output'))
        model.summary()
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(loss='mse', optimizer=Adam(lr=1e-4))
        model.compile(loss='mse', optimizer='Adam')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0001,
                                   patience=2,
                                   verbose=0,
                                   mode='auto')

        check_pointer = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=False,
                                        mode='auto',
                                        period=1)
        history_object = model.fit_generator(self.train_generator,
                                             samples_per_epoch=self.train_log.shape[0]*3,
                                             validation_data=self.validation_generator,
                                             nb_val_samples=self.valid_log.shape[0]*3,
                                             nb_epoch=50,
                                             callbacks=[check_pointer, early_stop],
                                             verbose=1)
        model.save('final_model.h5')
        with open('model.json', 'w') as f:
            json.dump(model.to_json(), f)
        self.plotting(history_object)
        test_x, test_y = self.prepare_test_data(self.test_log)
        print(np.array(test_x).shape, np.array(test_y).shape)
        evaluation = model.evaluate(np.array(test_x), np.array(test_y))
        print('test loss: {}'.format(evaluation))

    @staticmethod
    def plotting(history_object):
        print(history_object.history.keys())
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

    def continue_training(self):
        model = load_model('weights.28-0.0275.hdf5')

        model.fit_generator(self.train_generator,
                            samples_per_epoch=self.train_log.shape[0],
                            validation_data=self.validation_generator,
                            nb_val_samples=self.valid_log.shape[0],
                            nb_epoch=2,
                            verbose=1)
        model.save('final_model_2.h5')
        with open('final_model_2.json', 'w') as f:
            json.dump(model.to_json(), f)


def main():

    my_model = BaseModel()
    my_model.read_csv_data()
    my_model.create_generator()
    #my_model.train_model()
    my_model.continue_training()

if __name__ == "__main__":
    main()
