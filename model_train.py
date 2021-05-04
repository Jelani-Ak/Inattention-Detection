import os

import sns as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.applications.vgg16 import VGG16

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dataset = pd.read_csv(os.getcwd() + '/labels/driver_imgs_list.csv')
print(dataset.head(5))

src_path_train = "J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/train/"
src_path_test = "J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/test/"
image = cv2.imread('J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/test/mixed/img_143.jpg',
                   0)
print(image.shape)

class_poses = ['c0 Normal Driving',
               'c1 Texting - Right',
               'c2 Talking on the Phone - Right',
               'c3 Texting - Left',
               'c4 Talking on the Phone - Left',
               'c5 Operating the Radio',
               'c6 Drinking',
               'c7 Reaching Behind',
               'c8 Hair and Makeup',
               'c9 Talking to Passenger']

# Plot the first image of each class
for poses in class_poses:
    path = os.path.join(src_path_train, poses)
    for directory in os.listdir(src_path_train):
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            plt.imshow(img_array)
            plt.title(os.path.basename(poses))
            plt.show()
            break
        break

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.20)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)

train_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=6,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)
valid_generator = train_datagen.flow_from_directory(
    directory=src_path_train,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=6,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=42
)
test_generator = test_datagen.flow_from_directory(
    directory=src_path_test,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=2,
    class_mode=None,
    shuffle=False,
    seed=42
)


# Create the model
def prepare_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Dropout(0.5, input_shape=(60,)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5, input_shape=(60,)))
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(model.summary())
    return model


timeNow = time.strftime("%b-%d-%Y") + ' ' + time.strftime('%H %M %S', time.gmtime(12345))
name = "Inattention-Detection-{}".format(timeNow)
tensorboard_callback = TensorBoard(log_dir='logs/{}'.format(name))

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.05,
                              patience=5,
                              verbose=0,
                              min_lr=0.001)
model = prepare_model()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=5)
modelSummary = model.fit(train_generator,
                         validation_data=train_generator,
                         steps_per_epoch=train_generator.n // train_generator.batch_size,
                         validation_steps=valid_generator.n // valid_generator.batch_size,
                         callbacks=[reduce_lr, tensorboard_callback, early_stopping],
                         epochs=25)

score = model.evaluate(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

modelName = 'Inattention-Detection-Model'
model.save(os.getcwd() + '/exported_models/' + modelName + '_' + timeNow + '.h5')

plt.plot(modelSummary.history['accuracy'])
plt.plot(modelSummary.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.getcwd() + '/graphs/Accuracy - ' + modelName + '_' + timeNow + '.png')
plt.show()

plt.plot(modelSummary.history['loss'])
plt.plot(modelSummary.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.getcwd() + '/graphs/Loss - ' + modelName + '_' + timeNow + '.png')
plt.show()
