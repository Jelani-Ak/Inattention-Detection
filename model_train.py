import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import time
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential, Input
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# dataset = pd.read_csv(os.getcwd() + '/labels/driver_imgs_list.csv')
# print(dataset.head(5))

# model_summary = ''

train_directory = "J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/train/"
test_directory = "J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/test/"
# image = cv2.imread('J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/test/mixed/img_143.jpg', 0)
# print(image.shape)

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
    path = os.path.join(train_directory, poses)
    for directory in os.listdir(train_directory):
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
    directory=train_directory,
    target_size=(160, 120),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42
)
valid_generator = train_datagen.flow_from_directory(
    directory=train_directory,
    target_size=(160, 120),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=42
)
test_generator = test_datagen.flow_from_directory(
    directory=test_directory,
    target_size=(160, 120),
    color_mode="rgb",
    batch_size=12,
    class_mode=None,
    shuffle=False,
    seed=42
)


# def create_logs():
#     textfile = open("Inattention-Detection-{}".format(get_time()) + '.txt', "x")
#     textfile.write(prepare_model().summary())

# def prepare_model():
#     model_vgg16 = VGG16(weights='imagenet', include_top=False)
#     model_vgg16.summary()
#
#     # Create your own input format
#     keras_input = Input(shape=(224, 224, 3), name='image_input')
#
#     # Use the generated model
#     output_vgg16_conv = model_vgg16(keras_input)
#
#     # Add the fully-connected layers
#     x = Flatten(name='flatten')(output_vgg16_conv)
#     x = Dense(4096, activation='relu', name='fc1')(x)
#     x = Dense(4096, activation='relu', name='fc2')(x)
#     x = Dense(10, activation='softmax', name='predictions')(x)
#
#     # Create your own model
#     pretrained_model = Model(inputs=keras_input, outputs=x)
#     pretrained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     return pretrained_model


# # Bad model remove later
# def prepare_model():
#     my_model = Sequential()
#     my_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(160, 120, 3)))
#     my_model.add(Flatten())
#     my_model.add(Dense(128, activation='relu'))
#     my_model.add(Dense(10, activation='softmax'))
#     my_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
#     print(my_model.summary())
#     return my_model


# Create the model
def prepare_model():
    my_model = Sequential()
    my_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(160, 120, 3)))
    my_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    my_model.add(BatchNormalization())

    my_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', kernel_initializer='he_uniform'))
    my_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    my_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='he_uniform'))
    my_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    my_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='he_uniform'))
    my_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    my_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='he_uniform'))
    my_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    my_model.add(Flatten())
    my_model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    my_model.add(Dense(10, activation='softmax'))
    my_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    print(my_model.summary())
    return my_model


def get_time() -> str:
    return time.strftime("%b-%d-%Y") + ' ' + time.strftime('%H %M %S', time.localtime())


name = "Inattention-Detection-{}".format(get_time())
tensorboard_callback = TensorBoard(log_dir='logs/{}'.format(name))

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
                         # callbacks=[tensorboard_callback],
                         epochs=50)

score = model.evaluate(valid_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(get_time())

modelName = 'Inattention-Detection-Model [Bad Version]' + '-' + get_time()
# modelName = 'Inattention-Detection-Model' + '-' + get_time()
# modelName = 'Inattention-Detection-VGG16' + '-' + get_time()
model.save(os.getcwd() + '/exported_models/' + modelName + '.h5')

# Change it so that directories are created. The directories will take on the date and time as the name

plt.plot(modelSummary.history['accuracy'])
plt.plot(modelSummary.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.getcwd() + '/graphs/' + modelName + ' - Accuracy.png')
plt.show()

plt.plot(modelSummary.history['loss'])
plt.plot(modelSummary.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(os.getcwd() + '/graphs/' + modelName + ' - Loss.png')
plt.show()
