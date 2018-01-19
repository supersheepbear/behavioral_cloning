from preprocess import ProcessData
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD


class BaseModel(ProcessData):

    def __init__(self):
        ProcessData.__init__(self)
        self.batch_size = 256

    def train_data_prepare(self):
        self.load_pickle_data('input_train_data.p')

    def train_model(self):
        self.train_data_prepare()
        model = Sequential()
        model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(105, 200, 3)))
        model.add(Cropping2D(cropping=((29, 10), (0, 0))))
        model.add(Conv2D(24, 5, 5, activation="relu", name='conv1', subsample=(2, 2)))

        model.add(Conv2D(36, 5, 5, activation="relu", name='conv2', subsample=(2, 2)))
        model.add(Conv2D(48, 5, 5, activation="relu", name='conv3', subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3, activation="relu", name='conv4', subsample=(1, 1)))
        model.add(Conv2D(64, 3, 3, activation="relu", name='conv5', subsample=(1, 1)))
        model.add(Flatten(name='fl'))
        model.add(Dense(100, activation='relu', name='fc1'))
        model.add(Dropout(.5))
        model.add(Dense(50, activation='relu', name='fc2'))
        model.add(Dropout(.5))
        model.add(Dense(1, name='fc3'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer='adam')
        model.fit(self.x_train,
                  self.y_train,
                  batch_size=2048,
                  nb_epoch=3,
                  verbose=2,
                  validation_split=0.2,
                  shuffle=True)
        model.save('model.h5')


def train_model():
    my_model = BaseModel()
    my_model.train_model()


def main():
    train_model()


if __name__ == "__main__":
    main()
