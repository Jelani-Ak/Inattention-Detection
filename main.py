import os
import pandas as pd
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.models import load_model

dataset = pd.read_csv(os.getcwd() + '/labels/driver_imgs_list.csv')
print(dataset.head(5))

# class_poses = {'c0': 'Normal Driving',
#                'c1': 'Texting - Right',
#                'c2': 'Talking on the Phone - Right',
#                'c3': 'Texting - Left',
#                'c4': 'Talking on the Phone - Left',
#                'c5': 'Operating the Radio',
#                'c6': 'Drinking',
#                'c7': 'Reaching Behind',
#                'c8': 'Hair and Makeup',
#                'c9': 'Talking to Passenger'}

src_path_train = "J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/train/"
src_path_test = "J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/test/"
# image = cv.imread('J:/Jelani/Documents/Coding/Python [Extra]/Datasets/Distracted Driver/imgs/test/mixed/img_143.jpg', 0)
# print(image.shape)

# plt.figure(figsize=(12, 20))
# image_count = 1
# BASE_URL = src_path_train
# for directory in os.listdir(BASE_URL):
#     if directory[0] != '.':
#         for i, file in enumerate(os.listdir(BASE_URL + directory)):
#             if i == 1:
#                 break
#             else:
#                 fig = plt.subplot(5, 2, image_count)
#                 image_count += 1
#                 image = mpimg.imread(BASE_URL + directory + '/' + file)
#                 plt.imshow(image)
#                 plt.title(class_poses[directory])
#                 plt.show()

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


def prepare_model():
    activation = 'sigmoid'
    model = Sequential()
    model.add(Conv2D(32, 3, activation=activation, padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, activation=activation, padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(128, activation=activation, kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
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

model.save(os.getcwd() + '/exported_models/Inattention_1.h5')

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
# model = load_model('J:\Jelani\Documents\Coding\Python [Extra]\Models\Distracted Driving')
# model.summary()
# digit = model.predict_classes(img)
# print("Predicted Label : ", digit[0])
