import cv2
import pickle
import sklearn
import pandas as pd
import numpy as np
import scipy.misc
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class ProcessData:

    def __init__(self):
        self.x_train = []
        self.y_train = []

        self.driving_log = []
        self.x_train_processed = []
        self.x_train_input = []
        self.y_train_input = []
        self.x_valid_input = []
        self.y_valid_input = []
        self.x_test_input = []
        self.y_test_input = []
        self.train_generator = []
        self.valid_generator = []

    def split_data(self):
        print('splitting data')
        self.x_train, self.y_train = sklearn.utils.shuffle(self.x_train, self.y_train)
        self.x_train_input, self.x_valid_input, self.y_train_input, self.y_valid_input = train_test_split(self.x_train,
                                                                                                          self.y_train,
                                                                                                          test_size=0.2)

        self.x_train_input, self.x_test_input, self.y_train_input, self.y_test_input = train_test_split(self.x_train_input,
                                                                                                        self.y_train_input,
                                                                                                        test_size=0.1)
        print('x_train size:{}'.format(self.x_train_input.shape[0]))
        print('y_train size:{}'.format(self.y_train_input.shape))
        print('x_valid size:{}'.format(self.x_valid_input.shape[0]))
        print('y_valid size:{}'.format(self.y_valid_input.shape))
        print('x_test size:{}'.format(self.x_test_input.shape[0]))
        print('y_test size:{}'.format(self.y_test_input.shape))
        del self.x_train
        del self.y_train

    @staticmethod
    def get_batch(x, y, batch_size):
        num_samples = len(y)
        while 1:  # Loop forever so the generator never terminates
            for offset in range(0, num_samples, batch_size):
                x_batch = x[offset:offset + batch_size]
                y_batch = y[offset:offset + batch_size]
                yield sklearn.utils.shuffle(x_batch, y_batch)

    @staticmethod
    def adjust_images(image):
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        seq = iaa.Sequential([iaa.Multiply((0.8, 1.2), per_channel=0.2),
                              iaa.ContrastNormalization((0.75, 1.5)),
                              sometimes(iaa.Affine(shear=(-8, 8)))],
                             random_order=True)
        new_image = seq.augment_image(image)
        return new_image

    @staticmethod
    def yuv_convert(img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2YUV)

    def image_convert(self):
        new_x_train = []
        for img_index in range(self.x_train.shape[0]):
            img = self.x_train[img_index]
            img = img[70:140, :, :]
            img = self.adjust_images(img)
            #img = scipy.misc.imresize(img, (66, 200, 3))
            new_x_train.append(img)
            if (img_index+1) % 5000 == 0:
                print('pre process {}/{} images'.format(img_index+1, self.x_train.shape[0]+1))
        print('image convert done')
        print('x_train size:{}'.format(self.x_train.shape[0]))
        print('y_train size:{}'.format(self.y_train.shape))
        self.x_train = np.array(new_x_train)

    def read_csv_data(self):
        images = []
        driving_log = pd.read_csv(r'train_set_1_20\driving_log.csv', header=None)
        center_image_names = driving_log[0]
        measurement = driving_log[3]
        for index, image_path in center_image_names.iteritems():
            image = cv2.imread(image_path)
            images.append(image)
        self.x_train = np.array(images)
        self.y_train = np.array(measurement)

    @staticmethod
    def save_pickle_data(name, x_data, y_data):
        train_data = {'features': x_data,
                      'labels': y_data}
        with open(name, 'wb') as handle:
            pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle_data(self, name):
        train_file = name
        with open(train_file, mode='rb') as f:
            train_data = pickle.load(f)
        self.x_train = train_data['features']
        self.y_train = train_data['labels']


def image_process():
    my_image_process = ProcessData()
    my_image_process.load_pickle_data('aug_train_data.p')
    my_image_process.image_convert()
    my_image_process.split_data()
    my_image_process.save_pickle_data('input_train_data.p',
                                      my_image_process.x_train_input,
                                      my_image_process.y_train_input)
    my_image_process.save_pickle_data('input_valid_data.p',
                                      my_image_process.x_valid_input,
                                      my_image_process.y_valid_input)
    my_image_process.save_pickle_data('input_test_data.p',
                                      my_image_process.x_test_input,
                                      my_image_process.y_test_input)


def load_image_data():
    my_image_load = ProcessData()
    my_image_load.read_csv_data()
    my_image_load.save_pickle_data('train_data.p', my_image_load.x_train, my_image_load.y_train)


if __name__ == "__main__":
    #load_image_data()
    image_process()