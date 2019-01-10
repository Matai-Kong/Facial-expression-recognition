import sys
import os
import cv2
import time
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

import align.detect_face


minsize = 40 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

cap = cv2.VideoCapture(0)

# parameters for loading data and images
#image_path = sys.argv[1]

#detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
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


print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

# loading images
while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:
        gray_image = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)
        rgb_image = frame
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')


        bounding_boxes, landmarks = align.detect_face.detect_face(rgb_image, minsize, pnet, rnet, onet, threshold, factor)

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
        start2=time.time()
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

        end2=time.time()
        text=('一帧图像的网络处理时间为：{}'.format(end2-start2))

        text1=('The network processing time of one frame is:{}'.format(end2-start2))
        cv2.putText(rgb_image, text1, (10, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 0),2)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        cv2.imshow('iframe', bgr_image)

        #bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('../images/predicted_test_image.png', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('../images/p1.png', rgb_image)
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
