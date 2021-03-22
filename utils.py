import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
import yaml


def count_samples_per_class(X, Y):

    per_class_dict = {}
    for x, y in zip(X, Y):

        if not per_class_dict.__contains__(y):
            per_class_dict[y] = ["x"]
        else:
            per_class_dict[y].append(x)

    return per_class_dict


def load_samples(images_path, nb_classes=0, rdm_state=42):

    np.random.seed(rdm_state)

    data = {}
    for i, (root, dir, imgs) in tqdm(enumerate(os.walk(images_path))):
        if i == 0:
            continue

        label = int(root.split("/")[-1])

        data[label] = imgs
    labels = np.random.choice(list(data.keys()), nb_classes)

    if nb_classes == 0:
        return data

    filtered_data = {label: data[label] for label in labels}

    return filtered_data


def load_images(list, images_path, input_shape):

    loaded_imgs = []

    for img in tqdm(list):
        i = cv2.imread(os.path.join(images_path, img))
        i = cv2.resize(i, input_shape[:-1])
        loaded_imgs.append(i)

    return loaded_imgs


def enable_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def _transform_images(is_ccrop=False):
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (160, 160, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train
    return transform_images


def _transform_targets(y_train):
    return y_train


def _parse_tfrecord(binary_img=False, is_ccrop=False):
    def parse_tfrecord(tfrecord):
        if binary_img:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/filename': tf.io.FixedLenFeature([], tf.string),
                        'image/encoded': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        else:
            features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                        'image/img_path': tf.io.FixedLenFeature([], tf.string)}
            x = tf.io.parse_single_example(tfrecord, features)
            image_encoded = tf.io.read_file(x['image/img_path'])
            x_train = tf.image.decode_jpeg(image_encoded, channels=3)

        y_train = tf.cast(x['image/source_id'], tf.float32)

        x_train = _transform_images(is_ccrop=is_ccrop)(x_train)
        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train
    return parse_tfrecord


def load_tfrecord_dataset(tfrecord_name, batch_size,
                          binary_img=False, shuffle=True, buffer_size=10240,
                          is_ccrop=False):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(binary_img=binary_img, is_ccrop=is_ccrop),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)

    return loaded
