import os
import numpy as np
import pandas as pd
import imageio
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.utils.visualize_util import plot


def augment_side_camera_data(df):
    """
    augments data with left and right cameras, with adjustment for steering angle

    :param df: dataframe with left, right images and steering angles
    :return: dataframe with assumed center image and adjusted steering angle
    """

    df_left = df[["left", "steering"]]
    df_left.rename(columns={"left": "center"}, inplace=True)
    df_left["steering"] += 0.2
    df_right = df[["right", "steering"]]
    df_right.rename(columns={"right": "center"}, inplace=True)
    df_right["steering"] -= 0.2
    df_new = pd.concat([df_left, df_right], ignore_index=True)
    return df_new


def parse_driving_data(data_dir, all_cameras=False, ignore_no_steer=False):
    """
    parses driving lod data to extract train, validation and test sets.
    option to use images from all cameras, or ignore images with zero steering angle from training data.

    :param data_dir: directory containing driving log and images
    :param all_cameras: use all cameras?
    :param ignore_no_steer: ignore images with zero steering angle?
    :return: train, validation and test dataframes (shuffled)
    """

    def fix_img_path(df):
        """
        fix leading space or relative path in `center` columns (image path)

        :param df: driving-log dataframe
        :return:
        """
        df['center'] = df['center'].str.strip()
        df['center'] = df['center'].apply(lambda fpath:
                                          fpath if (fpath.startswith(data_dir) or fpath.startswith('/'))
                                          else os.path.join(data_dir, fpath))

    df = pd.read_csv(os.path.join(data_dir, "driving_log.csv"))

    # df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)

    if all_cameras:
        df_aug_train = augment_side_camera_data(df_train)
        df_train = pd.concat([df_aug_train, df_train], ignore_index=True)

        df_aug_valid = augment_side_camera_data(df_valid)
        df_valid = pd.concat([df_aug_valid, df_valid], ignore_index=True)

        # df_aug_test = augment_side_camera_data(df_test)
        # df_test = pd.concat([df_aug_test, df_test], ignore_index=True)

    if ignore_no_steer:
        df_train = df_train.loc[df_train["center"] != 0]
        df_valid = df_valid.loc[df_valid["center"] != 0]
        # df_test = df_test.loc[df_test["center"] != 0]

    df_train = shuffle(df_train)
    df_valid = shuffle(df_valid)
    # df_test = shuffle(df_test)

    fix_img_path(df_train)
    fix_img_path(df_valid)
    # fix_img_path(df_test)

    return df_train, df_valid#, df_test


def generate_data(df, batch_size, allow_flip=False):
    """
    creates generator of size `batch_size` over dataframe `df`

    :param df: input dataframe
    :param batch_size: batch size
    :param flip: randomly flip images horizontally
    :return: generator with images and labels
    """

    ix = df.index
    while True:
        ix = np.random.permutation(ix)
        for offset in range(0, df.shape[0], batch_size):
            ix_batch = ix[offset:offset + batch_size]
            lst_images, lst_labels = [], []
            flip_prob = np.random.rand(batch_size)
            for i, loc in enumerate(ix_batch):
                fpath = df.at[loc, 'center']
                img = imageio.imread(fpath)
                label = df.at[loc, 'steering']
                if allow_flip:
                    if flip_prob[i] > 0.5:
                        img = np.fliplr(img)
                        label *= -1
                lst_images.append(img)
                lst_labels.append(label)
            images = np.asarray(lst_images)
            labels = np.asarray(lst_labels)
            yield images, labels


def normalize(img):
    return img / 255. - 0.5


def rgb_to_grayscale(img):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(img)


def create_model():
    """
    lenet with batch-normalization and dropout

    :return: keras model
    """

    model = Sequential()
    model.add(Lambda(rgb_to_grayscale, input_shape=(160, 320, 3)))
    model.add(Lambda(normalize))
    model.add(Cropping2D(((70, 30), (0, 0))))
    model.add(Conv2D(6, 5, 5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(10, 5, 5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, 5, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


def main(model_save_path, data_dir, all_cameras, ignore_no_steer, nb_epoch, batch_size, weights_path):
    # df_train, df_valid, df_test = parse_driving_data(data_dir, all_cameras, ignore_no_steer)
    df_train, df_valid = parse_driving_data(data_dir, all_cameras, ignore_no_steer)

    generator_train = generate_data(df_train, batch_size, allow_flip=True)
    generator_valid = generate_data(df_valid, batch_size, allow_flip=True)
    # generator_test = generate_data(df_test, batch_size, allow_flip=True)

    model = create_model()
    if weights_path is not None:
        model.load_weights(weights_path)

    model.fit_generator(generator_train, samples_per_epoch=df_train.shape[0], nb_epoch=nb_epoch,
                        validation_data=generator_valid, nb_val_samples=df_valid.shape[0])
    # print("Test Accuracy", model.evaluate_generator(generator_test, val_samples=df_test.shape[0], nb_worker=1))

    model.save(model_save_path)
    plot(model, to_file='model.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train and evaluate model for car simulator')

    parser.add_argument('model_save_path', help="Path to save model")
    parser.add_argument('data_dir', help="Data directory")
    parser.add_argument('--all_cameras', dest='all_cameras', action='store_true')
    parser.add_argument('--ignore_no_steer', dest='ignore_no_steer', action='store_false')
    parser.add_argument('--nb_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weights_path', help="Path to weights", default=None)

    parser.set_defaults(all_cameras=False)

    args = parser.parse_args()

    main(args.model_save_path, args.data_dir, args.all_cameras, args.ignore_no_steer,
         args.nb_epoch, args.batch_size, args.weights_path)
