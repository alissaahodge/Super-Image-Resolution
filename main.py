#!/usr/bin/python
# Start by importing the libraries we need
import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import io

# face_images is a dataset of face images downloaded online that we will be using to train this model
face_images = glob.glob('lfw/**/*.jpg') #returns path of images
my_image_path = sys.argv[1]
def main():

    print(face_images[:2], len(face_images))
    with open('face_images_path.pickle', 'wb') as f:
        pickle.dump(face_images, f)

    p = Pool(10)
    img_array = p.map(read, face_images)
    # In order to save time in future, we store our image array img_array using the pickle library
    with open('img_array.pickle', 'wb') as f:
        pickle.dump(img_array, f)
    print(len(img_array))
    with open('img_array.pickle', 'rb') as f:
        img_array = pickle.load(f)
    # plt.imshow(img_array[100])

    # We will use train data to train our model and validation data will be used to evaluate the model
    all_images = np.array(img_array)
    # Split test and train data. all_images will be our output images
    train_x, val_x = train_test_split(all_images, random_state=32, test_size=0.2)
    train_x_px = []

# Here we are calling on the pixalate image function to lower the images resolution and take those in as input
    for i in range(train_x.shape[0]):
        temp = pixalate_image(train_x[i, :, :, :])
        train_x_px.append(temp)

    train_x_px = np.array(train_x_px)

    # get low resolution images for the validation set
    val_x_px = []

    for i in range(val_x.shape[0]):
        temp = pixalate_image(val_x[i, :, :, :])
        val_x_px.append(temp)

    val_x_px = np.array(val_x_px)
    # plt.imshow(train_x[100])
    # plt.imshow(train_x_px[100])

    # lets build the model No TPU
    # with strategy.scope():
    Input_img = Input(shape=(80, 80, 3))

    # encoding architecture
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
    x3 = MaxPool2D(padding='same')(x2)

    x4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
    x5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
    x6 = MaxPool2D(padding='same')(x5)

    encoded = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)
    # encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)

    # decoding architecture
    x7 = UpSampling2D()(encoded)
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
    x9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
    x10 = Add()([x5, x9])

    x11 = UpSampling2D()(x10)
    x12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
    x13 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
    x14 = Add()([x2, x13])

    # x3 = UpSampling2D((2, 2))(x3)
    # x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
    # x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    decoded = Conv2D(3, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)

    autoencoder = Model(Input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    autoencoder.summary()
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=1, mode='min')

    # Training our model
    model_checkpoint = ModelCheckpoint('superResolution_checkpoint3.h5', save_best_only=True)
    history = autoencoder.fit(train_x_px, train_x,
                              epochs=100,
                              validation_data=(val_x_px, val_x),
                              callbacks=[early_stopper, model_checkpoint])
    autoencoder = tf.keras.models.load_model('superResolution_checkpoint3.h5')
    predictions = autoencoder.predict(val_x_px)

    n = 4
    plt.figure(figsize=(20, 10))

    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(val_x_px[i + 20])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(predictions[i + 20])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
    # results = autoencoder.evaluate(val_x_px, val_x)
    # print('loss, accuracy', results)
    print("testing my image")
    img = image.load_img(my_image_path, target_size=(80, 80, 3))
    img = image.img_to_array(img)
    img = img / 255.
    img = pixalate_image(img)
    plt.imshow(img)
    plt.show()
    print("now the next one")
    input_array = np.array([img])
    predict = autoencoder.predict(input_array)
    plt.imshow(predict[0])
    plt.show()

def read(path):
    # here we use tqdm to show a progress bar for the work done
    progress = tqdm(total=len(face_images), position=0)
    # Here we are taking advantage of multiprocessing library provided in python for ease of execution to reduce the size of all images to 80 x 80 pixels.
    img = image.load_img(path, target_size=(80, 80, 3))
    img = image.img_to_array(img)
    img = img / 255.
    progress.update(1)
    return img

# we will distort our images and take it as an input images.
def pixalate_image(image, scale_percent=40):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    small_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # scale back to original size
    width = int(small_image.shape[1] * 100 / scale_percent)
    height = int(small_image.shape[0] * 100 / scale_percent)
    dim = (width, height)

    low_res_image = cv2.resize(small_image, dim, interpolation=cv2.INTER_AREA)

    return low_res_image


if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()
