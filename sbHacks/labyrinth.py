import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__": 
    img = cv2.imread('labyrinth.png',0)
    img = cv2.fastNlMeansDenoising(img,None,5,7,21)
    edges = cv2.Canny(img,70,80)
    
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    print(np.shape(edges))
    plt.show()
