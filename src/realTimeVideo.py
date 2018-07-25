import cv2 as cv
import datetime
import pexpect
from threading import Thread
from PIL import Image
import numpy as np
import base64

def analyze(pathToSave, saveName , imagePath , imageFolder , image):
    python27 = '/home/alireza/anaconda2/envs/sentiment/bin/python'
    #print(image)
    scirptToExe = './server.py'

    runScript = "{} {} {} {} {} {}".format(python27,
                                        scirptToExe,
                                        pathToSave,
                                        saveName,
                                        imagePath,
                                         image)


    output = pexpect.run(runScript)
    print(output)

#reading video
vcap = cv.VideoCapture("abc.mp4")
vcap.set(cv.CAP_PROP_POS_MSEC,7000)

success, image = vcap.read()

#specefic frame number
count = 73500

#for every 1000 frame
while (success) & (count<74000):
    time = datetime.datetime.now()
    imageName = "a" + str(time).replace(" ", "") + ".png"
    #    stra = base64.b85encode(image)
    #  print(len(str(stra)))

    imageFolder = str(time).replace(" ", "")
    # pil_img = cv.cvtColor(image ,cv.COLOR_RGB2GRAY )
    # w = Image.fromarray(image)
    # im_np = np.asarray(w)
    # w.show()

    cv.imwrite("../.rawImages/{}".format(imageName), image)
    print(str(image))
    thread = Thread(target=analyze,
        args=("../processedImages/", imageName , "../.rawImages/{}".format(imageName) ,
              imageFolder ,str(image)))

    thread.start()
    count+=1000
    vcap.set(cv.CAP_PROP_POS_MSEC, count)
    success,image = vcap.read()
    exit(0)

print('hi')
#hiii

