import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from glob import glob
import os

# The following function takes an image as an input and converts it into a feature segmented
# binary image 
def get_mask(img):
    
    #==========PREPROCESSING BEGIN===================================
    #Applying medianBlur and Gaussian blur to the obtained image using a kernel of size 5
    blur = cv2.medianBlur(img,5)
    blur = cv2.GaussianBlur(img,(5,5),0)
    # plt.imshow(blur,'gray')
    # plt.show()

    #Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #first CLAHE is done
    hist_eq = clahe.apply(blur)
    hist_eq = cv2.equalizeHist(hist_eq) #Global Histogram Equalization
    # plt.imshow(hist_eq,'gray')
    # plt.show()


    #===========FEATURE EXTRACTION BEGIN================================
    # Otsu's thresholding
    ret,th = cv2.threshold(hist_eq,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plt.imshow(th,'gray')
    # plt.show()

    # Morphological Transformation: CLOSING
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel, iterations = 2)
    closing = cv2.bitwise_not(closing) # Inverting the obtained image
    # plt.imshow(closing,'gray')
    # plt.show()

    #Floodfill the unwanted boundaries
    height, width = img.shape
    cv2.floodFill(closing,None,(0,0),0)
    cv2.floodFill(closing,None,(0,height-1),0)
    cv2.floodFill(closing,None,(width-1,height-1),0)
    cv2.floodFill(closing,None,(width-1,0),0)
    # plt.imshow(closing,'gray')
    # plt.show()

    
    result = closing
    ret, binary_map = cv2.threshold(result,127,255,0)
    # Obtaining the connected components
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    areas1 = np.asarray(stats[1:,cv2.CC_STAT_AREA]) # Areas of each white space is obtained

    final_result = np.zeros((labels.shape), np.uint8) # A mask is created
    for i in range(0, nlabels - 1):
        if areas1[i] ==np.max(areas1):   #keep only the largest area
            final_result[labels == i + 1] = 255

    #Post Processing
    blur = cv2.GaussianBlur(final_result,(13,13),0) # Finally iamge is smoothened again
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(closing,'gray')
    # plt.show()

    return thresh


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Segmentation of Gallbladder images')
    parser.add_argument('-i', '--img_path', type=str, default='img', required=True, help="Path for the image folder")
    parser.add_argument('-d', '--det_path', type=str, default='det', required=True, help="Path for the detected masks folder")

    args = parser.parse_args()

    img_files = sorted(glob(os.path.join(args.img_path, "*jpg")))


    for images in img_files:
        file_name = os.path.basename(images)
        img = cv2.imread(str(images),0)

        det_mask = get_mask(img)
        cv2.imwrite(os.path.join(args.det_path, file_name), det_mask)
        








