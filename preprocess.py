import cv2
import pickle
import sklearn
import pandas as pd
import numpy as np
import scipy.misc
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
        self.train_generator = []
        self.valid_generator = []

    def split_data(self):
        self.x_train_input, self.y_train_input, self.x_valid_input, self.y_valid_input = train_test_split(self.x_train,
                                                                                                          self.y_train,
                                                                                                          test_size=0.2)
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
    def yuv_convert(img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2YUV)

    def image_convert(self):
        new_x_train = []
        for img_index in range(self.x_train.shape[0]):
            img = self.yuv_convert(self.x_train[img_index])
            img = scipy.misc.imresize(img, (105, 200, 3))
            new_x_train.append(img)
            if (img_index+1) % 5000 == 0:
                print('pre process {}/{} images'.format(img_index+1, self.x_train.shape[0]+1))
        print('image convert done')
        self.x_train = np.array(new_x_train)

    def read_csv_data(self):
        images = []
        driving_log = pd.read_csv(r'../windows_sim/training_set/driving_log.csv', header=None)
        center_image_names = driving_log[0]
        measurement = driving_log[3]
        for index, image_path in center_image_names.iteritems():
            image = cv2.imread(image_path)
            images.append(image)
        self.x_train = np.array(images)
        self.y_train = np.array(measurement)

    def save_pickle_data(self, name):
        train_data = {'features': self.x_train,
                      'labels': self.y_train}
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
    my_image_process.save_pickle_data('input_train_data.p')


def load_image_data():
    my_image_load = ProcessData()
    my_image_load.read_csv_data()
    my_image_load.save_pickle_data('train_data.p')


if __name__ == "__main__":
    #load_image_data()
    image_process()