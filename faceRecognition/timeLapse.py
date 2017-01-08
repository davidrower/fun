#
# Purpose: Take time-lapse photos from webcamera
#          for facial recognition training
#
# Usage:   Press c to capture a picture
#          Press q to quit the program
#
# Author:  David Rower
# Date:    January 7, 2017
#

import cv2

if __name__ == "__main__":
    index = 1
    color = (0, 255, 0)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    videoCapture = cv2.VideoCapture(0)
    while True:
        ret, frame = videoCapture.read()
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('{}.png'.format(index),frame)
            print('{} pictures captured'.format(index))
            cv2.waitKey(1000)
            index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
