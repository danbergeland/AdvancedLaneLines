import imageio
import sys
from moviepy.editor import VideoFileClip
from lanefinding import find_lane_lines


if __name__ == "__main__":
    vidPath = sys.argv[1]
    clip1 = VideoFileClip(vidPath)
    newClip = clip1.fl_image(find_lane_lines) #NOTE: this function expects color images!!
    dstName = 'outputs/laneFind'+vidPath
    newClip.write_videofile(dstName, audio=False)
