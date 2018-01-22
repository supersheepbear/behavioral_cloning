import json
import preprocess
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


class BaseModel(preprocess.ProcessData):

    def __init__(self):
        preprocess.ProcessData.__init__(self)
        self.batch_size = 256

    def train_model(self):
        model = Sequential()
        model.add(Lambda(lambda x: (x - 127.5) / 127.5, input_shape=(66, 200, 3)))
        model.add(Conv2D(24, 5, 5, name='conv1', subsample=(2, 2)))
        model.add(ELU())
        model.add(Conv2D(36, 5, 5, name='conv2', subsample=(2, 2)))
        model.add(ELU())
        model.add(Conv2D(48, 5, 5, name='conv3', subsample=(2, 2)))
        model.add(ELU())
        model.add(Conv2D(64, 3, 3, name='conv4', subsample=(1, 1)))
        model.add(ELU())
        model.add(Conv2D(64, 3, 3, name='conv5', subsample=(1, 1)))
        model.add(ELU())
        model.add(Flatten(name='fc0'))
        model.add(Dense(1164, name='fc1'))
        model.add(Dense(100, name='fc2'))
        model.add(ELU())
        model.add(Dense(50, name='fc3'))
        model.add(ELU())
        model.add(Dense(10, name='fc4'))
        model.add(ELU())
        model.add(Dense(1, name='output'))
        model.summary()
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #model.compile(loss='mse', optimizer=Adam(lr=1e-4))
        model.compile(loss='mse', optimizer='Adam')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0001,
                                   patience=0,
                                   verbose=0,
                                   mode='auto')

        history = model.fit_generator(self.train_generator,
                                      samples_per_epoch=self.train_log.shape[0],
                                      validation_data=self.validation_generator,
                                      nb_val_samples=self.valid_log.shape[0],
                                      nb_epoch=30,
                                      verbose=1,
                                      callbacks=[early_stop])
        self.plotting(history)
        test_x, test_y = self.read_data(self.test_log)

        evaluation = model.evaluate(test_x, test_y)
        print('test loss: {}'.format(evaluation))
        model.save('model.h5')
        with open('model.json', 'w') as f:
            json.dump(model.to_json(), f)

    @staticmethod
    def plotting(history):
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()


def main():
    my_model = BaseModel()
    my_model.read_csv_data()
    my_model.create_generator()
    my_model.train_model()


if __name__ == "__main__":
    main()
