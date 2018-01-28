import cv2
import pandas as pd
import numpy as np
import scipy.misc
from imgaug import augmenters as iaa
from sklearn.utils import shuffle
from keras.callbacks import Callback


class WeightsSaver(Callback):
    def __init__(self, model, N):
        self.model = model
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save_weights(name)
        self.batch += 1


class ProcessData:

    def __init__(self):
        self.data_length = 0
        self.driving_log = pd.DataFrame()
        self.train_log = pd.DataFrame()
        self.valid_log = pd.DataFrame()
        self.test_log = pd.DataFrame()
        self.train_generator = 0
        self.validation_generator = 0
        self.batch_size = 64

    def image_generator(self, log, is_train):
        batch_size = self.batch_size
        num_samples = log.shape[0]
        while 1:  # Loop forever so the generator never terminates
            shuffle(log)
            for offset in range(0, num_samples, batch_size):
                batch_samples = log.iloc[offset:offset + batch_size, :]
                images = []
                angles = []
                for position in range(0, 3):
                    for index, row in batch_samples.iterrows():

                        # Randomly pick one of the image from left/mid/right camera for training
                        image_path = row[position]
                        if position == 0:
                            current_angle = float(row[3])
                        if position == 1:
                            current_angle = float(row[3]) + 0.2
                        if position == 2:
                            current_angle = float(row[3]) - 0.2
                        image = cv2.imread(image_path)
                        # Apply pre processing
                        image, current_angle = self.image_process(image, current_angle)
                        images.append(image)
                        angles.append(current_angle)

                # trim image to only see section with road
                x_data = np.array(images)
                y_data = np.array(angles)
                yield shuffle(x_data, y_data)

    def create_generator(self):
        self.train_generator = self.image_generator(self.train_log, is_train=1)
        self.validation_generator = self.image_generator(self.valid_log, is_train=0)

    def split_data(self):
        split_index_1 = int(self.data_length * 0.75)
        split_index_1 = split_index_1 - split_index_1 % 64
        split_index_2 = int(self.data_length * 0.95)
        split_index_2 = split_index_2 - (split_index_2 - split_index_1) % 64
        self.train_log = self.driving_log.iloc[:split_index_1, :]
        self.valid_log = self.driving_log.iloc[split_index_1:split_index_2, :]
        self.test_log = self.driving_log.iloc[split_index_2:, :]

        print('train size:{}'.format(len(self.train_log) * 3))
        print('valid size:{}'.format(len(self.valid_log) * 3))
        print('test size:{}'.format(len(self.test_log) * 3))

    @staticmethod
    def shear(image, steering_angle, shear_range=50):
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        m = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, m, (cols, rows), borderMode=1)
        steering_angle += dsteering

        return image, steering_angle

    @staticmethod
    def gamma(image):

        gamma = np.random.uniform(0.8, 1.2)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def adjust_images(self, image, current_angle):

        cv2.add(image, np.array([np.random.uniform(-100, 100)]))
        image = self.gamma(image)
        image = cv2.resize(image, (128, 56))
        return image, current_angle

    def image_process(self, img, current_angle):
        # crop image
        img = img[50:140, :, :]
        # apply image augmentation techniques
        img, current_angle = self.adjust_images(img, current_angle)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2YUV)
        return img, current_angle

    def combine_csv(self):
        log_1 = pd.read_csv(r'track_2\round_1\driving_log.csv', header=None)
        log_2 = pd.read_csv(r'track_2\round_2\driving_log.csv', header=None)
        log_3 = pd.read_csv(r'track_2\round_3_reverse\driving_log.csv', header=None)
        log_4 = pd.read_csv(r'track_2\round_4_reverse\driving_log.csv', header=None)
        log_5 = pd.read_csv(r'track_2\difficult_curves\driving_log.csv', header=None)
        log_6 = pd.read_csv(r'track_2\round_4_reverse\driving_log.csv', header=None)
        log_7 = pd.read_csv(r'track_2\difficult_slope\driving_log.csv', header=None)
        self.driving_log = pd.concat([log_1, log_2])
        self.driving_log = pd.concat([self.driving_log, log_3])
        self.driving_log = pd.concat([self.driving_log, log_4])
        self.driving_log = pd.concat([self.driving_log, log_5])
        self.driving_log = pd.concat([self.driving_log, log_6])
        self.driving_log = pd.concat([self.driving_log, log_7])

    def read_csv_data(self):
        self.combine_csv()
        # shuffle driving_log
        self.driving_log = shuffle(self.driving_log)
        self.data_length = self.driving_log.shape[0]

        self.split_data()

    @staticmethod
    def prepare_test_data(log):
        center_image_names = log[0]
        angle = log[3]
        images = []
        for index, image_path in center_image_names.iteritems():
            image = cv2.imread(image_path)
            images.append(image)
        images = np.array(images)
        adjust_images = []
        for img in images:
            img = img[50:140, :, :]
            # resize image
            # change image from BGR to YUV
            image = cv2.resize(image, (128, 56))
            adjust_images.append(img)
        angle = np.array(angle)
        return adjust_images, angle
