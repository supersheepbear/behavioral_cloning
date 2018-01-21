import numpy as np
import preprocess
import pickle
import json
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


class BaseModel(preprocess.ProcessData):

    def __init__(self):
        preprocess.ProcessData.__init__(self)
        self.batch_size = 256

    def load_train_data(self):
        with open('input_train_data.p', mode='rb') as f:
            train_data = pickle.load(f)
        self.x_train_input = train_data['features']
        self.y_train_input = train_data['labels']
        print('train data size: {}'.format(self.x_train_input.shape))
        with open('input_valid_data.p', mode='rb') as f:
            valid_data = pickle.load(f)
        self.x_valid_input = valid_data['features']
        self.y_valid_input = valid_data['labels']
        print('valid data size: {}'.format(self.x_valid_input.shape))
        with open('input_test_data.p', mode='rb') as f:
            test_data = pickle.load(f)
        self.x_test_input = test_data['features']
        self.y_test_input = test_data['labels']
        print('test data size: {}'.format(self.x_test_input.shape))

    def yasen_model(self):
        self.load_train_data()
        img_h = 66
        img_w = 200
        n_channel = 3
        input_shape = (img_h, img_w, n_channel)
        # size of pooling area for max pooling
        pool_size = (2, 2)

        # both Sequential model and Function API are used for practice
        # model is based on the NVIDIA's paper - "End to End Learning for Self-Driving Cars"
        # https://arxiv.org/pdf/1604.07316v1.pdf
        # add convolution layers to abstract feature map
        base_model = Sequential()
        base_model.add(Conv2D(24, 5, 5, activation='relu', input_shape=input_shape))
        base_model.add(MaxPooling2D(pool_size=pool_size))
        base_model.add(Conv2D(48, 3, 3, activation='relu'))
        base_model.add(MaxPooling2D(pool_size=pool_size))
        base_model.add(Conv2D(96, 3, 3, activation='relu'))
        base_model.add(MaxPooling2D(pool_size=pool_size))
        # add dropout layer to avoid over fitting
        base_model.add(Dropout(0.25))
        # The base model can be placed by other pre-trained models such as VGG16
        # use function API
        x = base_model.output
        # add the fully-connected layers
        x = Flatten(name='flatten')(x)
        x = Dense(1164, activation='relu', name='fc1')(x)
        x = Dense(100, activation='relu', name='fc2')(x)
        x = Dense(50, activation='relu', name='fc3')(x)
        x = Dense(10, activation='relu', name='fc4')(x)
        # It is a regression problem, so only 1 output
        predictions = Dense(1, name='predictions')(x)
        model = Model(input=base_model.input, output=predictions)
        # print model architecture
        model.summary()
        # use "Adam" optimizer and mean square error for regression problem
        model.compile(optimizer='adam', loss='mse')
        # set training parameters
        nb_epoch = 3
        batch_size = 32

        # training the model
        model.fit(self.x_train_input, self.y_train_input,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            validation_data=(self.x_valid_input, self.y_valid_input))
        # evaluation
        evaluation = model.evaluate(self.x_test_input, self.y_test_input)
        print('test loss: {}'.format(evaluation))
        model.save('model.h5')
        with open('model.json', 'w') as f:
            json.dump(model.to_json(), f)

    def train_model(self):
        self.load_train_data()
        model = Sequential()
        model.add(Lambda(lambda x: (x - 128.) / 128, input_shape=(85, 320, 3)))
        model.add(Conv2D(24, 5, 5, name='conv1', subsample=(2, 2), W_regularizer=l2(0.001)))
        model.add(ELU())
        model.add(Conv2D(36, 5, 5, name='conv2', subsample=(2, 2), W_regularizer=l2(0.001)))
        model.add(ELU())
        model.add(Conv2D(48, 5, 5, name='conv3', subsample=(2, 2), W_regularizer=l2(0.001)))
        model.add(ELU())
        model.add(Conv2D(64, 3, 3, name='conv4', subsample=(1, 1), W_regularizer=l2(0.001)))
        model.add(ELU())
        model.add(Conv2D(64, 3, 3, name='conv5', subsample=(1, 1), W_regularizer=l2(0.001)))
        model.add(ELU())
        model.add(Flatten(name='fc0'))
        model.add(Dense(1164, name='fc1', W_regularizer=l2(0.001)))
        model.add(Dropout(.5))
        model.add(Dense(100, name='fc2', W_regularizer=l2(0.001)))
        model.add(ELU())
        model.add(Dropout(.5))
        model.add(Dense(50, name='fc3', W_regularizer=l2(0.001)))
        model.add(ELU())
        model.add(Dropout(.5))
        model.add(Dense(10, name='fc4', W_regularizer=l2(0.001)))
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
        print('valid data size: {}'.format(self.x_valid_input.shape))
        print(self.y_valid_input.shape)
        model.fit(self.x_train_input,
                  self.y_train_input,
                  batch_size=64,
                  nb_epoch=30,
                  verbose=2,
                  shuffle=True,
                  callbacks=[early_stop],
                  validation_data=(self.x_valid_input, self.y_valid_input))
        evaluation = model.evaluate(self.x_test_input, self.y_test_input)
        print('test loss: {}'.format(evaluation))
        model.save('model.h5')
        with open('model.json', 'w') as f:
            json.dump(model.to_json(), f)

    def continue_train_model(self):
        self.load_train_data()
        model = load_model('model.h5')
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0001,
                                   patience=3,
                                   verbose=0,
                                   mode='auto')
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd)
        model.fit(self.x_train_input,
                  self.y_train_input,
                  batch_size=512,
                  nb_epoch=30,
                  verbose=2,
                  shuffle=True,
                  validation_data=(self.x_valid_input, self.y_valid_input),
                  callbacks=[early_stop])
        evaluation = model.evaluate(self.x_test_input, self.y_test_input)
        print('test loss: {}'.format(evaluation))
        model.save('model_final.h5')


def train_model():
    my_model = BaseModel()
    my_model.train_model()


def continue_training():
    my_model = BaseModel()
    my_model.continue_train_model()


def main():
    train_model()
    #continue_training()


if __name__ == "__main__":
    main()
