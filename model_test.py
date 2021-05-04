import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

class_poses = ['Normal Driving',
               'Texting - Right',
               'Talking on the Phone - Right',
               'Texting - Left',
               'Talking on the Phone - Left',
               'Operating the Radio',
               'Drinking',
               'Reaching Behind',
               'Hair and Makeup',
               'Talking to Passenger']


def prepare(filepath):
    image_size = 224
    image_array = cv2.imread(filepath)
    new_array = cv2.resize(image_array, (image_size, image_size))
    return new_array.reshape(-1, image_size, image_size, 3)


model = tf.keras.models.load_model(os.getcwd() + '/exported_models/Inattention_2.h5')

prediction = model.predict(
    [prepare('J:/Jelani/Documents/Coding/Python [Extra]/Datasets/[Dataset] Unique Distracted Driver/img_24.jpg')])
print(class_poses[int(prediction[0][0])])
print(prediction.shape)
print(prediction)
