import cv2 
import numpy as np


def get_sobel_xy(img, sobel_kernel):
    sobx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobel_kernel) 
    soby = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobel_kernel) 
    return sobx, soby

def threshold_bitmask(img, scale8bit=True, thresh=(0,255)):
    scaled = img
    if scale8bit:
        scaled = np.uint8(255*img/np.max(img))
    bMask=np.zeros_like(scaled)
    bMask[(scaled>thresh[0])&(scaled<thresh[1])] = 1
    return bMask

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sob = cv2.Sobel(gray,cv2.CV_64F,int(orient=='x'),int(orient=='y'),ksize=sobel_kernel)
    sob = np.absolute(sob)
    return threshold_bitmask(sob,thresh=thresh)

def mag_thresh(img, sobel_kernel=3, thresh = (0,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sx, sy = get_sobel_xy(gray, sobel_kernel)
    mag = np.sqrt(sx**2+sy**2)
    return threshold_bitmask(mag,thresh=thresh)

def dir_thresh(img, sobel_kernel=3, thresh=(0,255)):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sx,sy =  get_sobel_xy(gray,sobel_kernel)
    dirs = np.arctan2(np.absolute(sy),np.absolute(sx))
    return threshold_bitmask(dirs,scale8bit=False,thresh=thresh)

def hsv_color_thresh(img,h_thresh=(0,180), s_thresh=(0,255), v_thresh=(0,255)):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    out = np.zeros_like(hsv[:,:,0])
    hout = np.zeros_like(out)
    hout[(hsv[:,:,0]<=h_thresh[1])&(hsv[:,:,0]>=h_thresh[0])] = 1
    sout = np.zeros_like(out)
    sout[(hsv[:,:,1]<=s_thresh[1])&(hsv[:,:,1]>=s_thresh[0])] = 1
    vout = np.zeros_like(out)
    vout[(hsv[:,:,2]<=v_thresh[1])&(hsv[:,:,2]>=v_thresh[0])] = 1
    out[(hout==1)&(sout==1)&(vout==1)]=1
    return out

def hls_color_thresh(img,h_thresh=(0,180), l_thresh=(0,255), s_thresh=(0,255)):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    out = np.zeros_like(hsv[:,:,0])
    hout = np.zeros_like(out)
    hout[(hsv[:,:,0]<=h_thresh[1])&(hsv[:,:,0]>=h_thresh[0])] = 1
    sout = np.zeros_like(out)
    sout[(hsv[:,:,2]<=s_thresh[1])&(hsv[:,:,2]>=s_thresh[0])] = 1
    lout = np.zeros_like(out)
    lout[(hsv[:,:,1]<=l_thresh[1])&(hsv[:,:,1]>=l_thresh[0])] = 1
    out[(hout==1)&(sout==1)&(lout==1)]=1
    return out

def rgb_color_thresh(img, r_thresh=(0,255), g_thresh=(0,255), b_thresh=(0,255)):
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    out = np.zeros_like(rgb[:,:,0])
    rout = np.zeros_like(out)
    rout[(rgb[:,:,0]<=r_thresh[1])&(rgb[:,:,0]>=r_thresh[0])] = 1
    gout = np.zeros_like(out)
    gout[(rgb[:,:,1]<=g_thresh[1])&(rgb[:,:,1]>=g_thresh[0])] = 1
    bout = np.zeros_like(out)
    bout[(rgb[:,:,2]<=b_thresh[1])&(rgb[:,:,2]>=b_thresh[0])] = 1
    out[(rout==1)&(gout==1)&(bout==1)]=1
    return out
    
