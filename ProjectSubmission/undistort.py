import pickle
import numpy as np
import cv2

# Load the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = pickle.load(open( "camera_cal/cal_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

defImgSize = (1280,720)
height = defImgSize[1]
width = defImgSize[0]

split = .068
#horizon from top (higher shows less)
redHorz = .64

roadPts = np.float32([[width*(.5-split),height*redHorz],[width*(.5+split),height*redHorz],[width*.99,height],[width*.01,height]])

defOverheadSize = (1280,720)
offset = 180
left = offset
right = defOverheadSize[0]-offset
top =  0
bot = defOverheadSize[1]

showPts =np.float32([[left,top],[right,top],[right,bot],[left,bot]])
defM = cv2.getPerspectiveTransform(roadPts, showPts)
invM = cv2.getPerspectiveTransform(showPts, roadPts)

def get_undistort(img):
    return cv2.undistort(img, mtx, dist, None, mtx)

def overhead_perspective(img,M=defM,img_size=defOverheadSize):
    return cv2.warpPerspective(img, M, img_size)

def revert_from_overhead(img,M=invM, img_size=defImgSize):
    return cv2.warpPerspective(img,M, img_size)

if __name__ == "__main__":

    filename ='camera_cal/calibration10.jpg'
    img = cv2.imread(filename)
    dst = get_undistort(img)
    newName = filename.replace('.jpg','test.jpg')
    cv2.imwrite(newName,dst)
