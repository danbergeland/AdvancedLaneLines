#Defines the main pipeline for lanefinding
import numpy as np
import cv2
import sys
from undistort import get_undistort, overhead_perspective, revert_from_overhead
from thresholds import mag_thresh, abs_sobel_thresh, hls_color_thresh, rgb_color_thresh
from lane_data_types import Lane, LaneHistory

LH = LaneHistory()
font = cv2.FONT_HERSHEY_SIMPLEX

def preprocess(img):
    smooth =  cv2.GaussianBlur(img, (3,3),0)
    return smooth

def postprocess(img):
    kernel = np.ones((11,11),np.uint8)
    close = cv2.morphologyEx(img,cv2.MORPH_CLOSE, kernel)
    return close

def make_binary_mask(img):
    #colorConversions
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    #magSobMask = abs_sobel_thresh(img,thresh=(40,255))
    #magSobMask = mag_thresh(img,thresh=(40,255))    
    rgbThresh = rgb_color_thresh(img,r_thresh=(185,255), g_thresh=(185,255), b_thresh=(185,255))
    hlsThresh = hls_color_thresh(img,h_thresh=(18,35), l_thresh=(0,255), s_thresh=(40,255))
    bMask = np.zeros_like(img[:,:,0])
    bMask[(rgbThresh==1)|(hlsThresh==1)]=1
    return bMask

def find_lane_lines(img):
    xDim = img.shape[1]
    yDim = img.shape[0]
    #convert to BGR, since MoviePy uses RGB, thresholds are written for BGR
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    #undistort the img
    img = get_undistort(img)    
    #create binary mask    
    msk = preprocess(img)
    msk = make_binary_mask(msk)
    #perspective transform, convert mask to black and white image
    road = overhead_perspective(msk*255)
    road = postprocess(road)

    ln = Lane()
    ln.ohroad = road
    #apply sliding windows
    ln.find_lines()
    #calc lane position
    ln.calc_lane_position()
    #calc road curvature 
    ln.calc_curves()
    
    
    #Use lane history to average curves and error detect
    LH.add_lane(ln)

    LH.update_stats()
    #LH.clean_lanes()

    #add color to overhead perspective
    LH.make_overlay(img)
    #perspective tranform the drawn lines to the roadway 
    newwarp = revert_from_overhead(LH.overlay_img) 

    #convert image back to RGB before merging
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    outImg = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    #add calc img for debugging
    cimg = ln.calcimg
    cw = 340
    ch = 200
    cimg = cv2.resize(cimg,(cw,ch))
    pipx = xDim-cw
    pipy = 0
    outImg[pipy:pipy+ch,pipx:pipx+cw] = cimg

    #add lay over text
    outImg = cv2.rectangle(outImg, (0,0),(700,105), (255,255,255), -1)
    curveText = "Curve Radius: " + str(int(LH.curve)) + 'm'
    outImg = cv2.putText(outImg,curveText, (10,65), font, 1,(0,0,0),2,cv2.LINE_AA)
    curveText = "Lane Pos. Right of Center : " + str(round(LH.position,3))+'m'
    outImg = cv2.putText(outImg,curveText, (10,30), font, 1,(0,0,0),2,cv2.LINE_AA)

    curveText = "Detection Quality: " + str(round(100*LH.quality)) + " %"
    outImg = cv2.putText(outImg,curveText, (10,100), font, 1,(0,0,0),2,cv2.LINE_AA)
    #curveText = "R dev: " + str(LH.right_std)
    #outImg = cv2.putText(outImg,curveText, (10,135), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    return outImg



if __name__ == "__main__":
    imgPath = sys.argv[1]
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    mask = 255*make_binary_mask(img)
    cv2.imwrite("outputs/BMASK.jpg", mask)
    img = find_lane_lines(img)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imwrite("outputs/OUTPUT.jpg",img)
    
