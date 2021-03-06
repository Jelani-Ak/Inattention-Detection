import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import cv2
import tensorflow as tf
from pathlib import Path

# Classes to detect
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

# Reshape image
def prepare(filepath):
    image_size = 160
    image_array = cv2.imread(filepath)
    new_array = cv2.resize(image_array, (image_size, image_size))
    return new_array.reshape(-1, image_size, image_size, 3)


model = tf.keras.models.load_model(os.getcwd() + '/exported_models/Inattention-Detection-Model-May-06-2021 18 00 51.h5')
pathlist = Path('/Datasets/Unique Distracted Driver/').glob('*.jpg')

# Predict every file in directory
for path in sorted(pathlist):
    path_in_str = str(path)
    prediction = model.predict([prepare(path_in_str)])
    print('[' + class_poses[int(prediction[0][0])] + '] ' + os.path.basename(path_in_str))
