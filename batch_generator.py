from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os


def normalize(img):
    return img / 255.


class HAARCascade:

    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def cut_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.3, 5)
        roi_color = None
        for (x, y, w, h) in faces:
            roi_color = img[y:y + h, x:x + w]
            break
        return roi_color


class BatchGenerator(Sequence):
    def __init__(self, X, config, shuffle=True, jitter=False):
        self._config = config.copy()

        self._images_folder_path = self._config['DATASET_PATH']
        self._cut_face = self._config['CUT_FACE']

        self._debug = self._config['DEBUG']

        self._shuffle = shuffle
        self._jitter = jitter

        self._haar = HAARCascade()

        self._images_paths = []
        self._labels = {}
        self._samples_per_label = {}

        for label, fnames in X.items():
            for fname in fnames:
                image = {'filename': fname}

                if label not in self._samples_per_label:
                    self._samples_per_label[label] = [fname]
                else:
                    self._samples_per_label[label].append(fname)

                if label not in self._labels:
                    self._labels[label] = len(self._labels)
                image['class'] = self._labels[label]
                self._images_paths.append(image)

        # augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),
                iaa.Flipud(0.2),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    rotate=(-5, 5),
                    shear=(-5, 5),
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 2),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),
                                   iaa.AverageBlur(k=(2, 7)),
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               # iaa.Invert(0.05, per_channel=True), # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               # iaa.Grayscale(alpha=(0.0, 1.0)),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        if shuffle:
            np.random.shuffle(self._images_paths)

    def _get_class_from_labels(self, idx):
        return list(self._labels.keys())[list(self._labels.values()).index(idx)]

    def __len__(self):
        return int(np.ceil(len(self._images_paths) / self._config['BATCH_SIZE']))

    def num_classes(self):
        return len(self._labels)

    def size(self):
        return len(self._images_paths)

    def __getitem__(self, idx):
        l_bound = idx * self._config['BATCH_SIZE']
        r_bound = (idx + 1) * self._config['BATCH_SIZE']

        if r_bound > len(self._images_paths):
            r_bound = len(self._images_paths)
            l_bound = r_bound - self._config['BATCH_SIZE']

        instance_count = 0
        a_batch = np.zeros((r_bound - l_bound, self._config['IMAGE_H'], self._config['IMAGE_W'], self._config['IMAGE_C']))
        p_batch = np.zeros((r_bound - l_bound, self._config['IMAGE_H'], self._config['IMAGE_W'], self._config['IMAGE_C']))
        n_batch = np.zeros((r_bound - l_bound, self._config['IMAGE_H'], self._config['IMAGE_W'], self._config['IMAGE_C']))

        if len(self._labels) > 2:
            y_batch = np.zeros((r_bound - l_bound, 3, len(self._labels)))
        else:
            y_batch = np.zeros((r_bound - l_bound, 3))

        for train_instance in self._images_paths[l_bound:r_bound]:

            # anchor image
            anchor_image = train_instance
            anchor_label = self._get_class_from_labels(anchor_image['class'])

            # positive
            positive_img_candidates = self._samples_per_label[anchor_label].copy()
            positive_img_candidates.remove(anchor_image['filename'])
            positive_image = np.random.choice(positive_img_candidates)

            # negative
            aux_label = self._labels.copy()
            aux_label.pop(anchor_label, None)
            negative_label = np.random.choice(list(aux_label.keys()))
            negative_image_candidates = self._samples_per_label[negative_label].copy()
            negative_image = np.random.choice(negative_image_candidates)

            # augment input image and fix object's position and size
            img_a = self.aug_image(anchor_image['filename'], label=anchor_label, jitter=self._jitter)
            img_p = self.aug_image(positive_image, label=anchor_label, jitter=self._jitter)
            img_n = self.aug_image(negative_image, label=negative_label, jitter=False)

            if self._debug:
                cv2.imshow("Anchor", img_a)
                cv2.imshow("Positive", img_p)
                cv2.imshow("Negative", img_n)
                cv2.waitKey(0)

            img_apn = [img_a, img_p, img_n]

            # assign input images to x_batch
            if normalize is not None:
                for i, img in enumerate(img_apn):
                    if len(img.shape) == 2:
                        img_apn[i] = img[..., np.newaxis]

                a_batch[instance_count] = normalize(img_apn[0])
                p_batch[instance_count] = normalize(img_apn[1])
                n_batch[instance_count] = normalize(img_apn[2])
            else:
                a_batch[instance_count] = img_apn[0]
                p_batch[instance_count] = img_apn[1]
                n_batch[instance_count] = img_apn[2]

            # increase instance counter in current batch
            instance_count += 1

        x_batch = {
            'anchor': a_batch,
            'anchorPositive': p_batch,
            'anchorNegative': n_batch
        }

        return x_batch, [y_batch, y_batch, y_batch]

    def aug_image(self, train_instance, label, jitter):

        image_name = os.path.join(self._images_folder_path, str(label), train_instance)

        if self._config['IMAGE_C'] == 1:
            image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        elif self._config['IMAGE_C'] == 3:
            image = cv2.imread(image_name)
        else:
            raise ValueError("Invalid number of image channels.")

        # cut_face
        if self._cut_face:
            face_roi = self._haar.cut_face(image)
            if face_roi is not None:
                image = face_roi

        if jitter:
            image = self.aug_pipe.augment_image(image)

        return cv2.resize(image, (self._config['IMAGE_W'], self._config['IMAGE_H']))

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._images_paths)

    @property
    def labels(self):
        return self._labels

    @property
    def samples_per_label(self):
        return self._samples_per_label

    @property
    def jitter(self):
        return self._jitter

    @property
    def shuffle(self):
        return self._shuffle

