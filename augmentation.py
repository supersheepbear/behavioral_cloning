import cv2
import model
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from sklearn.utils import shuffle


class DataAugment(model.BaseModel):
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_train_aug = []
        self.y_train_aug = []

    def read_original_images(self):
        print('reading images')
        model.BaseModel.load_pickle_data(self, 'train_data.p')

    def save_aug_images(self):
        print('saving images')
        self.x_train_aug, self.y_train_aug = shuffle(self.x_train_aug, self.y_train_aug)
        self.x_train_aug = self.x_train_aug[0:80000]
        self.y_train_aug = self.y_train_aug[0:80000]
        print(self.x_train_aug.shape)
        print(self.y_train_aug.shape)
        aug_train_data = {'features': self.x_train_aug,
                          'labels': self.y_train_aug}
        del self.x_train_aug
        del self.y_train_aug
        with open('aug_train_data.p', 'wb') as handle:
            pickle.dump(aug_train_data, handle, protocol=4)

    def add_side_data(self):
        print('adding side images')
        left_images = []
        right_images = []
        driving_log = pd.read_csv(r'train_set_1_20\driving_log.csv', header=None)
        left_image_names = driving_log[1]
        right_image_names = driving_log[2]
        measurement = driving_log[3]
        for index, image_path in left_image_names.iteritems():
            image = cv2.imread(image_path)
            left_images.append(image)
        correction = 0.2
        self.x_train_aug = np.concatenate((self.x_train_aug, np.array(left_images)), axis=0)
        self.y_train_aug = np.concatenate([self.y_train_aug, np.array(measurement) + correction])
        del left_images
        for index, image_path in right_image_names.iteritems():
            image = cv2.imread(image_path)
            right_images.append(image)
        self.x_train_aug = np.concatenate((self.x_train_aug, np.array(right_images)), axis=0)
        self.y_train_aug = np.concatenate([self.y_train_aug, np.array(measurement) - correction])
        del self.x_train
        del self.y_train

    def flip_images(self):
        print('flipping images')
        flip_seq = iaa.Sequential([iaa.Fliplr(1.0)])

        aug_images = flip_seq.augment_images(self.x_train)

        self.x_train_aug = np.concatenate((self.x_train, aug_images), axis=0)
        self.y_train_aug = np.concatenate([self.y_train, self.y_train * (-1)])


def image_augmentation():
    my_data_augment = DataAugment()
    my_data_augment.read_original_images()
    my_data_augment.flip_images()
    my_data_augment.add_side_data()
    my_data_augment.save_aug_images()


if __name__ == "__main__":
    image_augmentation()