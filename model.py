import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


class BaseModel:

    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.driving_log = []

    def read_csv_data(self):
        images = []
        driving_log = pd.read_csv(r'../windows_sim/training_set/driving_log.csv', header=None)
        center_image_names = driving_log[0]
        measurement = driving_log[3]
        for index, image_path in center_image_names.iteritems():
            image = plt.imread(image_path)
            images.append(image)
        self.x_train = np.array(images)
        self.y_train = np.array(measurement)

    def save_pickle_data(self):
        train_data = {'features': self.x_train,
                      'labels': self.y_train}
        with open('train_data.p', 'wb') as handle:
            pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle_data(self):
        train_file = 'train_data.p'
        with open(train_file, mode='rb') as f:
            train_data = pickle.load(f)
        self.x_train = train_data['features']
        self.y_train = train_data['labels']

    def load_aug_pickle_data(self):
        train_file = 'aug_train_data.p'
        with open(train_file, mode='rb') as f:
            train_data = pickle.load(f)
        self.x_train = train_data['features']
        self.y_train = train_data['labels']

    def train_model(self):
        model = Sequential()
        model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160, 320, 3)))
        model.add(Convolution2D(6, 5, 5, activation="relu"))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Dense(84))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.fit(self.x_train,
                  self.y_train,
                  validation_split=0.2,
                  shuffle=True,
                  nb_epoch=2,
                  verbose=2)
        model.save('model.h5')


class DataAugment(BaseModel):
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_train_aug = []
        self.y_train_aug = []

    def read_original_images(self):
        BaseModel.load_pickle_data(self)

    def save_aug_images(self):
        aug_train_data = {'features': self.x_train_aug,
                          'labels': self.y_train_aug}
        with open('aug_train_data.p', 'wb') as handle:
            pickle.dump(aug_train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def flip_images(self):
        flip_seq = iaa.Sequential([iaa.Fliplr(0.5)])
        self.x_train_aug = self.x_train
        self.y_train_aug = self.y_train
        aug_images = flip_seq.augment_images(self.x_train)
        self.x_train_aug = np.concatenate((self.x_train_aug, aug_images), axis=0)
        self.y_train_aug = np.array(list(self.y_train_aug) + list(self.y_train * (-1)))
        print(self.x_train_aug.shape)
        print(self.y_train_aug.shape)


def process_data():
    my_model = BaseModel()
    my_model.read_csv_data()
    my_model.save_pickle_data()


def image_augmentation():
    my_data_augment = DataAugment()
    my_data_augment.read_original_images()
    my_data_augment.flip_images()
    my_data_augment.save_aug_images()


def train_model():
    my_model = BaseModel()

    my_model.load_aug_pickle_data()
    #my_model.load_pickle_data()
    my_model.train_model()


def main():
    #process_data()
    #image_augmentation()
    train_model()


if __name__ == "__main__":
    main()
