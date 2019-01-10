import sys
import os
import cv2
from keras.models import load_model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input

from utils.detect_face import create_mtcnn
from utils.detect_face import detect_face


minsize = 40 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


# parameters for loading data and images
image_path = sys.argv[1]
#detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
#gender_model_path = '../trained_models/gender_mini_XCEPTION.21-0.95.hdf5'
#gender_model_path = '../trained_models/gender_models/gender_mini_XCEPTION.19-0.95.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
gender_offsets = (20, 20)
emotion_offsets = (20,40)


# loading models
#face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

print('Creating networks and loading parameters')
gpu_memory_fraction=1.0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = create_mtcnn(sess, './')

bounding_boxes, landmarks = detect_face(rgb_image, minsize, pnet, rnet, onet, threshold, factor)
nrof_faces = bounding_boxes.shape[0]#人脸数目
print('找到人脸数目为：{}'.format(nrof_faces))

crop_faces=[]
for face_position in bounding_boxes:
    face_position=face_position.astype(int)
    print(face_position[0:4])
    #cv2.rectangle(rgb_image, (face_position[0]-10, face_position[1]-10), (face_position[2]+10, face_position[3]+10), (0, 255, 0), 2)
  #  crop=rgb_image[face_position[1]:face_position[3], face_position[0]:face_position[2],]
#print(bounding_boxes)

bounding_a =  bounding_boxes.astype(int)
#print(bounding_a)
crop_faces =bounding_a[:, :-1]
#print(crop_faces)
crop_faces[:,2] = crop_faces[:,2]-crop_faces[:,0]
crop_faces[:,3] = crop_faces[:,3]-crop_faces[:,1]
print(crop_faces)

#faces = detect_faces(face_detection, gray_image)

for face_coordinates in crop_faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    print(x1,x2,y1,y2)
    gray_face = gray_image[y1:y2, x1:x2]
    print(gray_face.shape)
    print(gray_face)

    try:
        rgb_face = cv2.resize(rgb_face, (gender_target_size))
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue
#    print(rgb_face.shape)
#    print(rgb_face)
#    print('!!!!')
    rgb_face = preprocess_input(rgb_face, False)

    rgb_face = np.expand_dims(rgb_face, 0)

    gender_prediction = gender_classifier.predict(rgb_face)
    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]
#    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#    print('!!!!')
    gray_face = preprocess_input(gray_face, True)

    gray_face = np.expand_dims(gray_face, 0)

    gray_face = np.expand_dims(gray_face, -1)

    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]

    if gender_text == gender_labels[0]:
        color = (200, 0, 0)
    else:
        color = (0, 0, 200)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, gender_text, color, 0, -8, 0.8, 2)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -30, 0.8, 2)

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite('../images/predicted_test_image.png', bgr_image)
