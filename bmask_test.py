#Defines the main pipeline for lanefinding
import numpy as np
import cv2
import sys
from undistort import get_undistort, overhead_perspective
from thresholds import abs_sobel_thresh, mag_thresh, dir_thresh, hsv_color_thresh, hls_color_thresh, rgb_color_thresh


if __name__ == "__main__":
    imgPath = sys.argv[1]
    img = cv2.imread(imgPath)
    
    #sobMask = abs_sobel_thresh(img,thresh=(20,200))
    #cv2.imwrite("outputs/sobX.jpg", 255*sobMask)
    #magSobMask = mag_thresh(img,thresh=(60,255))  
    #cv2.imwrite("outputs/magSob.jpg",255*magSobMask)
    #dirThreshR = dir_thresh(img, sobel_kernel=3, thresh=(.7,1.3))
    #cv2.imwrite("outputs/dirThreshR.jpg",255*dirThreshR)
    #dirThreshL = dir_thresh(img, sobel_kernel=3, thresh=(1.0,1.2))
    #cv2.imwrite("outputs/dirThreshL.jpg",255*dirThreshL)
    #RL = np.zeros_like(dirThreshL)
    #RL[(dirThreshR==1) & (magSobMask==1)] = 1
    #cv2.imwrite("outputs/RL.jpg",255*RL)
    #raw yellow h value ~60, or 45:75 range
    #print(cv2.cvtColor(np.uint8([[[255,255,0]]]),cv2.COLOR_RGB2HSV))
    #yellow filters:
    #hsvThresh = hsv_color_thresh(img, h_thresh=(20,40), s_thresh=(10,255), v_thresh=(0,255))
    #cv2.imwrite("outputs/hsvThresh.jpg",255*hsvThresh)
    #hlsThresh = hls_color_thresh(img,h_thresh=(20,40), l_thresh=(10,255), s_thresh=(10,255))

    rgbThresh = rgb_color_thresh(img,r_thresh=(185,255), g_thresh=(185,255), b_thresh=(185,255))
    hlsThresh = hls_color_thresh(img,h_thresh=(18,35), l_thresh=(0,255), s_thresh=(40,255))
    bMask = np.zeros_like(img[:,:,0])
    bMask[(rgbThresh==1)|(hlsThresh==1)]=1
    cv2.imwrite("outputs/bMask.jpg",255*bMask)
    kernel = np.ones((11,11),np.uint8)
    close = cv2.morphologyEx(255*bMask,cv2.MORPH_CLOSE, kernel)
    cv2.imwrite("outputs/bMaskClosed.jpg",close)
    oh =overhead_perspective(close)
    cv2.imwrite("outputs/bMaskOH.jpg",oh)


