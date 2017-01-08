'''
  Purpose: Generates a set of training images to be
           used for facial recognition
 
  Usage:   python trainingSet.py <label name>
 
  Author:  David Rower
  Date:    January 7, 2017
'''

from glob import glob
import sys, os
import cv2
import numpy as np

# facial detection using haarcascades from openCV
def detectFaces(image,faceCascade):
    flag = cv2.cv.CV_HAAR_SCALE_IMAGE
    faces = faceCascade.detectMultiScale(
                        image,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(60, 60),
                        flags = flag)
    return faces

# returns directory, and lists of xml and image files
def checkDirectory():
    fullPath  = os.path.realpath(__file__)
    directory = os.path.dirname(fullPath)
    people    = glob("{}/*/".format(directory))
    cascade   = glob("{}/haarcascade_frontalface_*.xml".format(directory))
    images, extensions = {}, ['jpg','png']
    for person in people: 
        imageList = []
        for extension in extensions: 
            imageList.extend(glob("{0}/*.{1}".format(person,extension)))
        images["{}".format(person)] = imageList
    return directory, cascade, images

# iterate through image path names, append faces to a training set
def createTrainingSet(images,faceCascade):
    index, trainingSet, labels = 0, [], []
    for person, imageList in images.iteritems():
        for imagePath in imageList:
            color = cv2.imread(imagePath)
            gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            gray  = cv2.equalizeHist(gray)
            faces = detectFaces(gray,faceCascade)  
#            print("Faces found: in {0}: {1}".format(imagePath,len(faces)))
            for (x,y,w,h) in faces: 
                cv2.imshow('Face', gray[y:y+h,x:x+w])
                cv2.waitKey(100)
                trainingSet.append(gray[y:y+h,x:x+w])
                labels.append(index)
        index += 1
    return trainingSet, labels

if __name__ == "__main__":       
    # find directory and important file paths
    directory, cascade, images = checkDirectory()
    names = [key.split('/')[-2] for key in images.keys()]
    print("Directory: {}/".format(directory))
    print("People: {}".format(names))

    # quits if no cascade or image files are found
    if not cascade: 
        sys.exit("No cascade classifier file found.")
    if not images: 
        sys.exit("No images found.")

    # creates the cascade
    faceCascade = cv2.CascadeClassifier(cascade[0])
    print("Cascade used: {}".format(cascade[0]))

    # creates the training set given dictionary of image file paths
    trainingSet, labels = createTrainingSet(images,faceCascade)
    for label in np.unique(labels): 
        imageNumber = labels.count(label)
        print("{}: {} training images".format(names[label],imageNumber))
    cv2.destroyAllWindows()

    faceRecognizer = cv2.createLBPHFaceRecognizer()
    faceRecognizer.train(trainingSet,np.array(labels))

    color = cv2.imread(images[0])
    print("Test on {}".format(images[0]))
    gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = detectFaces(gray,faceCascade) 
    for (x,y,w,h) in faces: 
        prediction, confidence = faceRecognizer.predict(gray[y:y+h,x:x+w])
        print("Prediction: {0}, Confidence: {1}".format(prediction,confidence))









             

                
            
