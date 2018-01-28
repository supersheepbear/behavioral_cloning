import cv2
import pandas as pd
import numpy as np
from sklearn.utils import shuffle


class ProcessData:
    """This Class is responsible for pre process images
    """
    def __init__(self):
        """Initiations
        """
        self.data_length = 0
        self.driving_log = pd.DataFrame()
        self.train_log = pd.DataFrame()
        self.valid_log = pd.DataFrame()
        self.test_log = pd.DataFrame()
        self.train_generator = 0
        self.validation_generator = 0
        self.batch_size = 64

    def image_generator(self, log, is_train):
        """A generator to provide images and labels for model batches
        :param log: The combined driving log to provide model input
        :param is_train: Indicate if this generator is for training or
        validation. Only use center images for validation.
        :return: data sets and labels for model input
        """
        batch_size = self.batch_size
        num_samples = log.shape[0]
        while 1:  # Loop forever so the generator never terminates
            shuffle(log)
            for offset in range(0, num_samples, batch_size):
                batch_samples = log.iloc[offset:offset + batch_size, :]
                images = []
                angles = []
                for index, row in batch_samples.iterrows():
                    # Only use center images in validation data
                    if is_train == 1:
                        position = np.random.randint(0, 3)
                    else:
                        position = 0
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

                x_data = np.array(images)
                y_data = np.array(angles)
                yield shuffle(x_data, y_data)

    def create_generator(self):
        """Created generator for training batch and validation batch
        """
        self.train_generator = self.image_generator(self.train_log, is_train=1)
        self.validation_generator = self.image_generator(self.valid_log, is_train=0)

    def split_data(self):
        """Split data to train, validation, test sets
        """
        split_index_1 = int(self.data_length * 0.75)
        split_index_1 = split_index_1 - split_index_1 % 64
        split_index_2 = int(self.data_length * 0.95)
        split_index_2 = split_index_2 - (split_index_2 - split_index_1) % 64
        self.train_log = self.driving_log.iloc[:split_index_1, :]
        self.valid_log = self.driving_log.iloc[split_index_1:split_index_2, :]
        self.test_log = self.driving_log.iloc[split_index_2:, :]

        print('train size:{}'.format(len(self.train_log)))
        print('valid size:{}'.format(len(self.valid_log)))
        print('test size:{}'.format(len(self.test_log)))

    @staticmethod
    def shear(image, steering_angle, shear_range=50):
        """Apply random shear for images
        :param image: image array from cv2 imread function
        :param steering_angle: angle
        :param shear_range: the amount of maximum shear value for x axis
        :return sheared image and angle value
        """
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

    def adjust_images(self, image, current_angle):
        """apply random brightness/random flip and random shear to images
        :param image: image array from cv2 imread function
        :param current_angle: angle value from driving log
        :return: adjusted image and angle value
        """
        cv2.add(image, np.array([np.random.uniform(-100, 100)]))
        image, current_angle = self.shear(image, current_angle)
        flip_flag = np.random.randint(0, 2)
        if flip_flag == 1:
            image = np.fliplr(image)
            current_angle = current_angle *(-1.)

        return image, current_angle

    def image_process(self, img, current_angle):
        """Pre process images before fed into model"""
        # crop image
        img = img[70:130, :, :]
        # apply image augmentation techniques
        img, current_angle = self.adjust_images(img, current_angle)
        # apply GaussianBlur
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # change image from BGR to YUV
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2YUV)
        return img, current_angle

    def combine_csv(self):
        """Combine multiple driving logs to training on multiple data sets
        """
        log_1 = pd.read_csv(r'my_data\pure_left\driving_log.csv', header=None)
        log_2 = pd.read_csv(r'udacity_data\driving_log.csv', header=None)
        log_3 = pd.read_csv(r'my_data\pure_right\driving_log.csv', header=None)
        log_4 = pd.read_csv(r'my_data\left_recover\driving_log.csv', header=None)
        log_5 = pd.read_csv(r'my_data\right_recover\driving_log.csv', header=None)
        log_6 = pd.read_csv(r'my_data\keyboard_pure_left\driving_log.csv', header=None)
        log_7 = pd.read_csv(r'e:\course\self-driving\my_projects\windows_sim\training_set\driving_log.csv', header=None)
        log_8 = pd.read_csv(r'my_data\bridge_and_straight\driving_log.csv', header=None)
        log_9 = pd.read_csv(r'my_data\left_curve\driving_log.csv', header=None)
        log_10 = pd.read_csv(r'my_data\right_curve\driving_log.csv', header=None)

        self.driving_log = pd.concat([log_1, log_2])
        self.driving_log = pd.concat([self.driving_log, log_3])
        self.driving_log = pd.concat([self.driving_log, log_4])
        self.driving_log = pd.concat([self.driving_log, log_5])
        self.driving_log = pd.concat([self.driving_log, log_6])
        self.driving_log = pd.concat([self.driving_log, log_7])
        self.driving_log = pd.concat([self.driving_log, log_8])
        self.driving_log = pd.concat([self.driving_log, log_9])
        self.driving_log = pd.concat([self.driving_log, log_10])

    def read_csv_data(self):
        """Read driving log csv files
        """
        self.combine_csv()
        # shuffle driving_log
        self.driving_log = shuffle(self.driving_log)
        self.data_length = self.driving_log.shape[0]

        self.split_data()

    @staticmethod
    def prepare_test_data(log):
        """prepare test sets data for our model
        :param log: driving log to provide test data
        """
        center_image_names = log[0]
        angle = log[3]
        images = []
        for index, image_path in center_image_names.iteritems():
            image = cv2.imread(image_path)
            images.append(image)
        images = np.array(images)
        adjust_images = []
        for img in images:
            img = img[70:130, :, :]
            # resize image
            # change image from BGR to YUV
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2YUV)
            adjust_images.append(img)
        angle = np.array(angle)
        return adjust_images, angle
