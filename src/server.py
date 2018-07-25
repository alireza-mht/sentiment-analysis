import os
from pymongo import MongoClient
from pymongo import cursor
from image_emotion_gender_demo import sentiment
from pprint import pprint
import datetime
from utils.inference import load_image
import cv2 as cv
import sys


pathToSave = sys.argv[1]
predictedImageName = sys.argv[2]
imagepath = sys.argv[3]
imageFolder = sys.argv[4]
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     "+ pathToSave  + "          !!!!!!!!!!!!")

#
# pathToSave = "../processedImages/"
# predictedImageName = "1.jpg"
# imagepath = "/home/alireza/faceClassificationProjcet/face_classification/1.jpg"
# imageFolder="1"



def getDbNumber():
    a = db.sentiments.aggregate([{
        "$group": {
            "_id": "",
            "totalSad": {
                "$sum": "$sad"
            },
            "totalHappy": {
                "$sum": "$happy"
            },
            "totalSurprise": {
                "$sum": "$surprise"
            },
            "totalFear": {
                "$sum": "$fear"
            },
            "totalDisgust": {
                "$sum": "$disgust"
            },
            "totalNetural": {
                "$sum": "$neutral"
            },
            "totalAngry": {
                "$sum": "$angry"
            },
        }
    }])
    count = 0;
    for doc in a:
        print(doc)


sentimentData=  sentiment(pathToSave , predictedImageName , imagepath ,imageFolder )

    #os.remove(imagepath)
# remove the predictedImage

#
# data = sentimentData['data']
# client = MongoClient()
# client.database_names()
# db = client.admin
# sentiment={
#     "sad":data['sad'],
#     "angry":data['angry'],
#     "neutral":data['neutral'],
#     "disgust":data['disgust'],
#     "fear":data['fear'],
#     "happy":data['happy'],
#     "surprise":data['surprise'],
#     "time":imageFolder
# }
#
#
# sentiments= db.sentiments
# post_id = sentiments.insert_one(sentiment).inserted_id

#if you want numbers in database
#getDbNumber()


#serverStatusResult=db.command("serverStatus")
#pprint(serverStatusResult)
