import cv2 
import numpy as np
from collections import deque

class Line():
    def __init__(self):
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #number of pixels 
        #(left pixels)/(
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float') 
        #x values for plotted curve fitting
        self.allx = None  
        #y values for plotted curve fitting
        self.ally = None
        self.detected = False

    def calc_curvature(self):
        ym_per_pix = 30/720 #meters per pixel y dim        
        xm_per_pix = 3.7/700 #meters per pixel x dim
        y_eval = np.max(self.ally)
        line_fit = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
        if line_fit[0]!=0:
            self.radius_of_curvature = ((1 + (2*line_fit[0]*y_eval*ym_per_pix + line_fit[1])**2)**1.5) / np.absolute(2*line_fit[0])
        


#Create a lane class.  The purpose of this object is to hold the lane line info, and do operations that apply to both lines
class Lane():
    def __init__(self):
        self.leftLine = Line()
        self.rightLine = Line()
        self.lanePosition = 0
        self.laneCurvature = 0
        #save overhead masked image.
        self.ohroad = []
        self.calcimg = []
        #generalized, so if a camera were mounted off-center.  0 for this project.
        self.cameraOffset = 0
        self.laneWidthMeters = 3.7

    def find_lines(self):
        img = self.ohroad
        height = img.shape[0]
        width = img.shape[1]
        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[int(height/2):,:],axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 12
        # Set height of windows
        window_height = np.int(height/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = height - (window+1)*window_height
            win_y_high = height - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        fity = np.linspace(0, height-1, height )

        #Describe very few pixels as inadequate to 'detect' the lane
        lane_inds_threshold = 700

        # Extract left and right line pixel positions
        if len(left_lane_inds)>lane_inds_threshold:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            left_fit = np.polyfit(lefty, leftx, 2)
            self.leftLine.current_fit = left_fit
            self.leftLine.ally = fity
            fit_leftx = left_fit[0]*fity**2 + left_fit[1]*fity + left_fit[2]
            self.leftLine.allx = fit_leftx
            self.leftLine.line_base_pos = fit_leftx[-1]
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            self.leftLine.detected = True


        if len(right_lane_inds)>lane_inds_threshold:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 
            right_fit = np.polyfit(righty, rightx, 2)
            self.rightLine.current_fit = right_fit
            self.rightLine.ally = fity
            fit_rightx = right_fit[0]*fity**2 + right_fit[1]*fity + right_fit[2]
            self.rightLine.allx = fit_rightx
            self.rightLine.line_base_pos = fit_rightx[-1]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            self.rightLine.detected = True
        self.calcimg = out_img


    def calc_lane_position(self):
        #must run 'find_lines'
        #get the center value
        if self.leftLine.detected ==True and self.rightLine.detected==True:
            center = int(self.calcimg.shape[1]/2)
            left = abs(self.leftLine.line_base_pos - center)
            right = abs(self.rightLine.line_base_pos - center)
            #add camera offset.  Take ratio of left/left+right-.5 to get center lane = 0.  Multiply by lane width.
            self.lanePosition = self.cameraOffset + (left/(left+right)-.5)*self.laneWidthMeters

    def calc_curves(self):
        if self.leftLine.detected ==True and self.rightLine.detected==True:
            self.rightLine.calc_curvature()
            self.leftLine.calc_curvature()
            curves = 0
            curveSum = 0
            if self.rightLine.radius_of_curvature != 0:
                curves+=1
                curveSum += self.rightLine.radius_of_curvature
            if self.leftLine.radius_of_curvature != 0:
                curves+=1
                curveSum += self.leftLine.radius_of_curvature
            if curves > 0:
                self.laneCurvature = curveSum/curves
                
        

#Create a history class.  The purpose of this object is to allow averaging or other time variant data operations.
class LaneHistory():
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.curve = 0
        self.position = 0
        self.left_fit = [np.array([False])]  
        self.quality = 0
        self.left_std = [np.array([False])]
        self.right_fit = [np.array([False])]  
        self.right_std = [np.array([False])]
        self.overlay_img = []
        self.draw_dims = (720,1280)
        
    #lanes will be added to the left (index 0) and shift through to maxlen
    def add_lane(self, lane):
        #add lanes where both lines are detected.  This rejects frames with partial lines
        if lane.leftLine.detected==True and lane.rightLine.detected==True:
            self.buffer.appendleft(lane)

    def update_stats(self):
        curvesum = 0
        pos = 0
        left_coeffs = np.array([1,1,1])
        right_coeffs = np.array([1,1,1])
        for lane in self.buffer:
            curvesum += lane.laneCurvature
            pos += lane.lanePosition
            left_coeffs = np.vstack((left_coeffs,lane.leftLine.current_fit))
            right_coeffs = np.vstack((right_coeffs,lane.rightLine.current_fit))
        if len(self.buffer) > 0:
            self.curve = curvesum/len(self.buffer)        
            self.position = pos/len(self.buffer)    
            #drop the initialization coeffs
            left_coeffs = left_coeffs[1:,:]
            right_coeffs = right_coeffs[1:,:]
            #average the values in the columns to get 3 coefficients
            self.left_std = np.std(left_coeffs, axis=0)
            self.right_std = np.std(right_coeffs, axis=0)
            #calculate quality.             
            left_dev = np.asarray(self.left_std)
            right_dev = np.asarray(self.right_std)
            if right_dev.all != 0 and left_dev.all != 0: 
                StdDif = np.divide((right_dev - left_dev),(right_dev+left_dev))
                self.quality = 1-np.average(np.absolute(StdDif))
            
            self.left_fit = np.average(left_coeffs, axis=0)
            self.right_fit = np.average(right_coeffs, axis=0)

    def make_overlay(self, img):
        # Create an image to draw the lines on
        color_warp = np.zeros_like(img).astype(np.uint8)
        # Generate fit lines based on all lanes in the buffer
        fity = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = self.left_fit[0]*fity**2 + self.left_fit[1]*fity + self.left_fit[2]
        right_fitx = self.right_fit[0]*fity**2 + self.right_fit[1]*fity + self.right_fit[2]
        
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, fity]))]).astype(np.int)
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))]).astype(np.int)
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, [pts], (0,255, 0))
        self.overlay_img = color_warp

    def clean_lanes(self):
        #create standard deviation comparison
        left_dev = np.asarray(self.left_std)
        right_dev = np.asarray(self.right_std)
        if right_dev.all != 0 and left_dev.all != 0:
            #if the std are similar, they will be near 0, else they will be near 1 or greater than 1
            StdDif = np.divide((right_dev - left_dev),(right_dev+left_dev))
            #if any of the stdDif are greater than .9, one of the lines can't be trusted
            if np.greater([.99,.99,.99],np.absolute(StdDif)).any:
                #figure out which line it is
                #if average negative, left dev is greater (left line is bad)
                if np.average(StdDif) < 0:
                    #copy the right line over to left, except for intercept
                    self.left_fit[0] = self.right_fit[0]
                    self.left_fit[1] = self.right_fit[1]

                #else the right line is wrong
                else:
                    self.right_fit[0] = self.left_fit[0]
                    self.right_fit[1] = self.left_fit[1]
           
             
        

        #for lane in self.buffer:


            #if the left is lost, but not the right, use right line curve data
            #TODO:
            #if left_lost and !right_lost:
            
            #TODO: if right line is lost, use left.

            #TODO: if lines are fine, update expected intercept based on self.intercept_lr
  
