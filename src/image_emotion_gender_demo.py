import sys
import os
import datetime
import json
import zipfile
import shutil

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.inference import load_image
from utils.preprocessor import preprocess_input
from PIL import Image


# helper functions and utils
def has_attribute(data, attribute):
    '''
    description: check if the dictionary object given contains the key attribute
    :param data: data object to look up for the given attribute
    :param attribute: given attribute
    :return: return false if data doesn't contain the attribute key and return true otherwise
    '''
    return attribute in data and data[attribute] is not None

def incrementEmotionTag(number, tag, jsonObject):
    '''
    description: increments the number of count of the given tag by the given number
    :param number: given number to add up to tag's count
    :param tag: given tag to increment its count
    :param jsonObject: json object which contains the tag
    :return: None
    '''
    if (not has_attribute(jsonObject, tag)):
        jsonObject[tag] = 0

    jsonObject[tag] += number

def addPictureReference(imageName, tag, jsonObject):
    '''
    description: adds the picture name to the array which stores which picture classifies as what emotion
    :param imageName: given image name
    :param tag: indicates which emotion tag the given imageName belongs
    :param jsonObject: json object
    :return: None
    '''
    key = '{}-people'.format(tag)

    if (not has_attribute(jsonObject, key)):
        jsonObject[key] = []

    jsonObject[key].append(imageName)

def zipdir(path, ziph , predictedIamgeName):
    '''
    description: zips a directory with all of its sub files and folders
    :param path: path to the directory
    :param ziph: zip handler
    :return: None
    '''


    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))
            os.remove(root+'/'+file)
    ziph.write(os.path.join('','../'+predictedIamgeName))
    shutil.rmtree(path)
    #os.remove(os.path.join('','../'+predictedIamgeName))

def zipImages( imageFolder ,pathToSave, predictedImageName):
    # TODO: <Alipoor> truncate the folder then try to store the new images
    now = imageFolder
    zipf = zipfile.ZipFile((pathToSave+'/{}.zip').format(now), 'w', zipfile.ZIP_DEFLATED)
    zipdir('../'+imageFolder+'/', zipf , predictedImageName )
    zipf.close()
    return zipf

# parameters for loading data and images
def sentiment ( pathToSave , predictedImageName , image_path , imageFolder):
    detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
    gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')
    font = cv2.FONT_HERSHEY_SIMPLEX

    # hyper-parameters for bounding boxes shape
    gender_offsets = (30, 60)
    gender_offsets = (10, 10)
    emotion_offsets = (20, 40)
    emotion_offsets = (0, 0)

    # loading models
    face_detection = load_detection_model(detection_model_path)
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

    faces = detect_faces(face_detection, gray_image)

    imageId = 0

    # json data container
    data = {}
    os.makedirs('../'+imageFolder)
    for face_coordinates in faces:
        imageId = imageId + 1

        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        rgb_face = rgb_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        bgr_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2BGR)
        cv2.imwrite('../'+imageFolder+ '/{}.png'.format(imageId), bgr_face)
        try:
            rgb_face = cv2.resize(rgb_face, (gender_target_size))
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        rgb_face = preprocess_input(rgb_face, False)
        rgb_face = np.expand_dims(rgb_face, 0)
        gender_prediction = gender_classifier.predict(rgb_face)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]   # gender_text contains gender for each person detected!

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        temp = emotion_classifier.predict(gray_face)

        emotion_label_arg = np.argmax(temp)

        sad = (temp[0])[4]
        happy = (temp[0])[3]
        score= happy - sad
        print("sad:" + str(sad))
        print("happy:" + str(happy))

        # emotion_text contains emotion label for each person detected in the image!
        emotion_text = emotion_labels[emotion_label_arg]

        # increment emotion tag number
        incrementEmotionTag(1, emotion_text, data)

        # increment <emotion + gender> tag number
        genderEmotionTag = "{}-{}".format(emotion_text, gender_text)
        incrementEmotionTag(1, genderEmotionTag, data)

        # set which image has what label
        addPictureReference(imageId, emotion_text, data)

        if gender_text == gender_labels[0]:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)


        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 2)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -50, 1, 2)


    # TODO: <PR-Lab> use the given API to send json data and also send zipped files.
    data['date-time'] = str(predictedImageName)   # json data
    data = compeletData(data)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('../'+predictedImageName, bgr_image)
    faceImage = zipImages(imageFolder , pathToSave , predictedImageName)# zipping images
    return {'data':data , 'faceImage':faceImage ,
            'image':bgr_image  , 'faceNum': imageId }


#this method compelte data and set all number of attribute in data
def compeletData(data):
    # if not has_attribute(data, 'disgust-man'):
    #     data['disgust-man']=0
    # if not has_attribute(data, 'disgust-woman'):
    #     data['disgust-woman'] = 0
    # if not has_attribute(data, 'angry-man'):
    #     data['angy-man']=0
    # if not has_attribute(data, 'angry-woman'):
    #     data['angry-woman']=0
    if not has_attribute(data , 'sad' ):
        data['sad'] =0
    if not has_attribute(data , 'angry'):
        data['angry']=0
    if not has_attribute(data, 'disgust'):
        data['disgust']=0
    if not has_attribute(data, 'fear'):
        data['fear'] = 0
    if not  has_attribute(data , 'happy'):
        data['happy']=0
    if not has_attribute(data,'surprise'):
        data['surprise']=0
    if not has_attribute(data,'neutral'):
        data['neutral']=0
    return data