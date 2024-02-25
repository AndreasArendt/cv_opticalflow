import cv2
import numpy as np

from optical_flow.CalcFlow import CalcFlow
from optical_flow.BlockMatching import BlockMatching

BlockSize   = 64 # Block Size of chunks (must be gcd of width/height)
MaxMovement = 32 # maximum movement of rectangle from starting point

cap = cv2.VideoCapture('data/uav_2_resized.mp4')

for i in range(1,150):
    ret, frame = cap.read()       

HEIGHT = frame.shape[0]
WIDTH = frame.shape[1]

Start_x = int(WIDTH/2-BlockSize/2)
Start_y = int(HEIGHT/2-BlockSize/2)

while(ret):    
    previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for i in range(1,30):
        ret, frame = cap.read()

    x = Start_x
    y = Start_y

    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get previous subframe
    previous_subframe = previous_frame[x:x+BlockSize, y:y+BlockSize]
    
    # find previous subframe in current frame + Margin
    current_macroframe = current_frame[y-MaxMovement:y+BlockSize+MaxMovement, x-MaxMovement:x+BlockSize+MaxMovement]

    best_ssd, best_match_c, best_match_r = BlockMatching(current_macroframe, previous_subframe, BlockSize)

    # draw rectangles
    previous_frame_bgr = cv2.cvtColor(previous_frame, cv2.COLOR_BAYER_BG2BGR)
    current_frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_BAYER_BG2BGR)
        
    # draw rectangle of object in prev frame
    previous_frame_rect_frame = cv2.rectangle(previous_frame_bgr.copy(), (x, y),
                                        (x + BlockSize, y + BlockSize),
                                        [0, 0, 255], 1)       
    
    # draw rectangle of allowed movement in prev frame
    previous_frame_rect_frame = cv2.rectangle(previous_frame_rect_frame, (x-MaxMovement, y-MaxMovement),
                                        (x+BlockSize+MaxMovement, y+BlockSize+MaxMovement),
                                        [0, 255, 255], 1) 

    # draw rectangle of object in current frame
    current_frame_rect_frame = cv2.rectangle(current_frame_bgr.copy(), (best_match_c+x-MaxMovement, best_match_r+y-MaxMovement),
                                        (best_match_c + BlockSize + x - MaxMovement, best_match_r + BlockSize + y - MaxMovement),
                                        [0, 0, 255], 1)       

    # draw rectangle of allowed movement in current frame
    current_frame_rect_frame = cv2.rectangle(current_frame_rect_frame, (x-MaxMovement, y-MaxMovement),
                                        (x+BlockSize+MaxMovement, y+BlockSize+MaxMovement),
                                        [0, 255, 255], 1) 

    splot = np.hstack([previous_frame_rect_frame, current_frame_rect_frame])
    
    # Display the result
    cv2.imshow('Image', splot)
    cv2.waitKey(1)  

    print("ssd: " + str(best_ssd))

    a = 1
