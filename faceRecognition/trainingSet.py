'''
  Purpose: Generates a set of training images to be
           used for facial recognition
 
  Usage:   python trainingSet.py <label name>
 
  Author:  David Rower
  Date:    January 7, 2017
'''

from glob import glob
import sys, os
import cv2, Image
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
    cascade   = glob("{}/*.xml".format(directory))
    images, extensions = [], ['jpg','png']
    for extension in extensions: 
        images.extend(glob("{0}/*.{1}".format(directory,extension)))
    return directory, cascade, images

# iterate through image path names, append faces to a training set
def createTrainingSet(images,faceCascade):
    trainingSet = []
    for imagePath in images:
        color = cv2.imread(imagePath)
        gray  = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        faces = detectFaces(gray,faceCascade)  
        print("Faces found: {}".format(len(faces)))
        for (x,y,w,h) in faces: 
            cv2.rectangle(color, (x,y), (x+w,y+h), (100,255,0),2)
            cv2.imshow('Face', color)
            cv2.waitKey(100)
            trainingSet.append(cv2.equalizeHist(gray[x:x+w,y:y+h]))
    return trainingSet

if __name__ == "__main__":
    # find directory and important file paths
    directory, cascade, images = checkDirectory()
    print("Directory: {}/".format(directory))

    # quits if no cascade or image files are found
    if not cascade: 
        sys.exit("No cascade classifier file found.")
    if not images: 
        sys.exit("No images found.")

    # creates the cascade
    faceCascade = cv2.CascadeClassifier(cascade[0])

    print("Cascade used: {}".format(cascade[0]))
    print("Training set: {}".format(len(images)))
    
    label = raw_input("Label? ")
    trainingSet = createTrainingSet(images,faceCascade)
    cv2.destroyAllWindows()

    

             

                
            
