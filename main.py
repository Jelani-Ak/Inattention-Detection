import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.models import load_model

src_path_train = "J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/train/"
src_path_test = "J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/test/"
image = cv.imread('J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/test/mixed/img_143.jpg', 0)
print(image.shape)

# sub_class = os.listdir(src_path_train)
# sub_class = os.listdir(src_path_test)
#
# fig = plt.figure(figsize=(10, 5))
# path = os.path.join(src_path_test, sub_class[0])
# for i in range(4):
#     plt.subplot(240 + 1 + i)
#     img = plt.imread(os.path.join(src_path_test + '*.jpg'))
#     plt.imshow(img, cmap=plt.get_cmap('gray'))
#
# path = os.path.join(src_path_train, sub_class[1])
# for i in range(4, 8):
#     plt.subplot(240 + 1 + i)
#     img = plt.imread(os.path.join(src_path_train + '*.jpg'))
#     plt.imshow(img, cmap=plt.get_cmap('gray'))

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.15)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)
valid_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=8,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=42
)
test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=4,
    class_mode=None,
    shuffle=False,
    seed=42
)


def prepare_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='sigmoid', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='sigmoid', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5, input_shape=(60,)))
    model.add(Dense(512, activation='sigmoid', kernel_initializer='he_uniform'))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    print(model.summary())
    return model


def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.05,
                              patience=5,
                              verbose=0,
                              min_lr=0.001)
model = prepare_model()
modelSummary = model.fit(train_generator,
                         validation_data=train_generator,
                         steps_per_epoch=train_generator.n // train_generator.batch_size,
                         validation_steps=valid_generator.n // valid_generator.batch_size,
                         callbacks=[reduce_lr],
                         epochs=20)

score = model.evaluate(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(str(os.getcwd()) + 'J:\Jelani\Documents\Coding\Python [Extra]\Models\Distracted Driving\Inattention_1.h5')

plt.plot(modelSummary.history['accuracy'])
plt.plot(modelSummary.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(modelSummary.history['loss'])
plt.plot(modelSummary.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# img = load_image(image)
# model = load_model(str(os.getcwd()) + 'J:\Jelani\Documents\Coding\Python [Extra]\Models\Distracted Driving')
# model.summary()
# digit = model.predict_classes(img)
# print("Predicted Label : ", digit[0])
