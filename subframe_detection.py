import cv2
import numpy as np
import matplotlib.pyplot as plt

from optical_flow.CalcFlow import CalcFlow
from optical_flow.BlockMatching import BlockMatching

BlockSize   = 32 # Block Size of chunks (must be gcd of width/height)
MaxMovement = 32 # maximum movement of rectangle from starting point

#cap = cv2.VideoCapture('data/uav_2_resized.mp4')
cap = cv2.VideoCapture('data/slow_traffic_small_resized.mp4')

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

    best_ssd = np.Inf
    current_frame_off = np.Inf, np.Inf
    previous_frame_off = np.Inf, np.Inf
        
    dx_arr = np.array([])
    dy_arr = np.array([])
    dxy_arr = np.array([])
    theta_arr = np.array([])
    ssd_arr = np.array([])

    previous_frame_bgr = cv2.cvtColor(previous_frame, cv2.COLOR_BAYER_BG2BGR)
    current_frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_BAYER_BG2BGR)

    # iterate over each column, row in previous frame
    for c in range(0, WIDTH-MaxMovement+1, BlockSize):                    
        for r in range(0, HEIGHT-MaxMovement+1, BlockSize):        
            # limit row and column indices
            c_idx_st = np.max([c-MaxMovement, 0])
            c_idx_en = np.min([c+MaxMovement+BlockSize, WIDTH])

            r_idx_st = np.max([r-MaxMovement, 0])
            r_idx_en = np.min([r+MaxMovement+BlockSize, HEIGHT])

            macro_block = current_frame[r_idx_st:r_idx_en, c_idx_st:c_idx_en]
            prev_block = previous_frame[r:r+BlockSize, c:c+BlockSize]
                            
            # run block matching
            ssd, c_off, r_off = BlockMatching(macro_block, prev_block, BlockSize)
            
            if ssd < best_ssd and ssd > 0 or True:
                best_ssd = ssd
                current_frame_off = r_off+r_idx_st, c_off+c_idx_st
                previous_frame_off = r,c

            
            current_subframe  = current_frame[current_frame_off[0]:current_frame_off[0]+BlockSize, current_frame_off[1]:current_frame_off[1]+BlockSize]
            previous_subframe = previous_frame[previous_frame_off[0]:previous_frame_off[0]+BlockSize, previous_frame_off[1]:previous_frame_off[1]+BlockSize]
                
            # Calc Optical Flow
            u, v, flow_u, flow_v, Ix, Iy, It = CalcFlow(current_subframe, previous_subframe)      
            
            # draw rectangles
            # previous_frame_bgr = cv2.cvtColor(previous_frame, cv2.COLOR_BAYER_BG2BGR)
            # current_frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_BAYER_BG2BGR)
                
            # draw rectangle of object in prev frame
            previous_frame_rect_frame = cv2.rectangle(previous_frame_bgr, (c, r),
                                                (c + BlockSize, r + BlockSize),
                                                [0, 0, 255], 1)       
            
            # draw rectangle of allowed movement in prev frame
            #previous_frame_rect_frame = cv2.rectangle(previous_frame_rect_frame, (c_idx_st, r_idx_st), (c_idx_en-1, r_idx_en-1), [0, 255, 255], 1) 

            # draw rectangle of object in current frame
            current_frame_rect_frame = cv2.rectangle(current_frame_bgr, (current_frame_off[1], current_frame_off[0]),
                                                (current_frame_off[1] + BlockSize, current_frame_off[0] + BlockSize),
                                                [0, 0, 255], 1)       

            # draw rectangle of allowed movement in current frame
            #current_frame_rect_frame = cv2.rectangle(current_frame_rect_frame, (c_idx_st, r_idx_st), (c_idx_en-1, r_idx_en-1), [0, 255, 255], 1) 

            u_norm = int(u / (np.sqrt( u**2 + v**2 )) * 16)
            v_norm = int(v / (np.sqrt( u**2 + v**2 )) * 16)            
            arrow_x_st = current_frame_off[1] + BlockSize // 2
            arrow_y_st = current_frame_off[0] + BlockSize // 2
            arrow_x_en = current_frame_off[1] + BlockSize // 2 + v_norm
            arrow_y_en = current_frame_off[0] + BlockSize // 2 + u_norm

            current_frame_rect_frame = cv2.arrowedLine(current_frame_rect_frame, (arrow_x_st, arrow_y_st), (arrow_x_en, arrow_y_en), [0, 255, 0], 1)  

            splot = np.hstack([previous_frame_rect_frame, current_frame_rect_frame])
            
            # Display the result
            cv2.imshow('Image', cv2.resize(splot, (640*2, 640), interpolation=cv2.INTER_NEAREST ))
            cv2.waitKey(1)  

            dx = current_frame_off[1] - c
            dy = current_frame_off[0] - r
            dxy = np.sqrt(dx**2 + dy**2)
            theta = np.arccos(dy/dxy)
            print("dx: " + str(dx), " dy: " + str(dy), "dist: " +  str(dxy) + " theta: " + str(theta / np.pi * 180.0) )

            dx_arr = np.append(dx_arr, dx)
            dy_arr = np.append(dy_arr, dy)
            dxy_arr = np.append(dxy_arr, dxy)
            theta_arr = np.append(theta_arr, theta)
            ssd_arr = np.append(ssd_arr, ssd)

            print("u: " + str(u) + " v: " + str(v))
            print("ssd: " + str(best_ssd))

            a = 1   
    a = 1

    # plt.subplot(211)
    # plt.subplot(2, 1, 1)
    # plt.hist(dx_arr, density=True, label="dx")
    # plt.hist(dy_arr, density=True, label="dy")  
    # plt.legend(loc="upper left")
    # plt.ylabel('Probability')
    # plt.xlabel('Data')

    # plt.subplot(2, 1, 2)
    # plt.hist(ssd_arr)
    # plt.show()
    