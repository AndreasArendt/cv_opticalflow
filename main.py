import cv2
import numpy as np
import matplotlib.pyplot as plt
from optical_flow.CalcFlow import CalcFlow

from optical_flow.BlockMatching import BlockMatching

BlockSize   = 64 #80 # Block Size of chunks (must be gcd of width/height)
MaxMovement = 32 # maximum movement of rectangle from starting point

def main():

    # Capture video from file or camera
    #cap = cv2.VideoCapture('data/slow_traffic_small.mp4')
    # cap = cv2.VideoCapture('uav_1.mp4')
    cap = cv2.VideoCapture('data/uav_2_resized.mp4')
    #cap = cv2.VideoCapture('data/vid.mp4')

    #skip first frames
    for i in range(1,1599):
        ret, frame = cap.read()       

    # find blocks from prev_frame in current_frame
    ret, frame = cap.read()       
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    fcnt = 0

    u_arr = np.array([])
    v_arr = np.array([])
    elev_arr = np.array([])
    azi_arr = np.array([])
    ssd_arr = np.array([])

    HEIGHT = frame.shape[0]
    WIDTH = frame.shape[1]

    for ii in range(1,700):
        ret, frame = cap.read()       

        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Processing frame: " + str(ii))
        
        fcnt += 1

        if not ret:
            break        

        best_ssd = np.Inf
        current_frame_off = np.Inf, np.Inf
        previous_frame_off = np.Inf, np.Inf
       
        # iterate over each column, row in previous frame
        for c in range(0, WIDTH-MaxMovement+1, BlockSize):            
            for r in range(0, HEIGHT-MaxMovement+1, BlockSize):        
                # limit row and column indices
                c_idx_st = np.max([c-MaxMovement, 0])
                c_idx_en = np.min([c+MaxMovement+BlockSize, WIDTH])

                r_idx_st = np.max([r-MaxMovement, 0])
                r_idx_en = np.min([r+MaxMovement+BlockSize, HEIGHT])

                macro_block = current_frame[r_idx_st:r_idx_en, c_idx_st:c_idx_en]
                prev_block = prev_frame[r:r+BlockSize, c:c+BlockSize]
                              
                # run block matching
                ssd, c_off, r_off = BlockMatching(macro_block, prev_block, BlockSize)
                
                if ssd < best_ssd and ssd > 0:
                    best_ssd = ssd
                    current_frame_off = r_off+r_idx_st, c_off+c_idx_st
                    previous_frame_off = r,c
                 
        
        print(str(r_off) + ", " + str(c_off))

        current_subframe  = current_frame[current_frame_off[0]:current_frame_off[0]+BlockSize, current_frame_off[1]:current_frame_off[1]+BlockSize]
        previous_subframe = prev_frame[previous_frame_off[0]:previous_frame_off[0]+BlockSize, previous_frame_off[1]:previous_frame_off[1]+BlockSize]
                
        # Calc Optical Flow
        u, v, flow_u, flow_v, Ix, Iy, It, elevation, azimuth = CalcFlow(current_subframe, previous_subframe)        
                
        debug = True
        if debug == True:
            DebugPlot(current_frame, current_frame_off, prev_frame, previous_frame_off, Ix, Iy ,It)
            print(str(u) + ", " + str(v), ", ssd: " + str(ssd))

        u_arr = np.append(u_arr,u)
        v_arr = np.append(v_arr,v)
        azi_arr = np.append(azi_arr, azimuth)
        elev_arr = np.append(elev_arr, elevation)
        ssd_arr = np.append(ssd_arr, ssd)

        prev_frame = current_frame

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            plt.show()
            break

    # Turn off interactive mode and close the OpenCV windows
    cv2.destroyAllWindows()
    cap.release()

    combined_array = np.column_stack((u_arr, v_arr, azi_arr, elev_arr, ssd_arr))

    # Save the combined array to a CSV file
    np.savetxt('./out/out.csv', combined_array, delimiter=',', header='u_arr,v_arr, azimut, elevation, ssd', comments='')

last_It = None
def DebugPlot(current_frame, current_frame_off, prev_frame, previous_frame_off, Ix, Iy ,It):
    global last_It

    # DEBUG - start
    large_current_frame = cv2.resize(current_frame.copy(), (448, 448), interpolation=cv2.INTER_NEAREST)
    large_prev_frame = cv2.resize(prev_frame.copy(), (448, 448), interpolation=cv2.INTER_NEAREST)

    current_frame_bgr = cv2.cvtColor(large_current_frame, cv2.COLOR_BAYER_BG2BGR)
    previous_frame_bgr = cv2.cvtColor(large_prev_frame, cv2.COLOR_BAYER_BG2BGR)

    scale_factor = 448//current_frame.shape[0]

    # Scale rectangle coordinates
    scaled_current_frame_off = (current_frame_off[0] * scale_factor, current_frame_off[1] * scale_factor)
    scaled_previous_frame_off = (previous_frame_off[0] * scale_factor, previous_frame_off[1] * scale_factor)

    # Draw rectangles on the frames with scaled coordinates
    current_rect_frame = cv2.rectangle(current_frame_bgr.copy(), (scaled_current_frame_off[1], scaled_current_frame_off[0]),
                                    (scaled_current_frame_off[1] + BlockSize * scale_factor -1, scaled_current_frame_off[0] + BlockSize * scale_factor - 1),
                                    [0, 0, 255], 1)
    previous_rect_frame = cv2.rectangle(previous_frame_bgr.copy(), (scaled_previous_frame_off[1], scaled_previous_frame_off[0]),
                                        (scaled_previous_frame_off[1] + BlockSize * scale_factor -1, scaled_previous_frame_off[0] + BlockSize * scale_factor - 1),
                                        [0, 255, 0], 1)


    if last_It is None:
        last_It = It.copy()

    current_large_It = cv2.resize(It.copy(), (448, 448), interpolation=cv2.INTER_NEAREST)
    previous_large_It  = cv2.resize(last_It.copy(), (448, 448), interpolation=cv2.INTER_NEAREST)

    current_It_bgr = cv2.cvtColor(current_large_It, cv2.COLOR_BAYER_BG2BGR)
    previous_It_bgr = cv2.cvtColor(previous_large_It, cv2.COLOR_BAYER_BG2BGR)

    current_It_rect_frame = cv2.rectangle(current_It_bgr, (scaled_current_frame_off[1], scaled_current_frame_off[0]),
                                        (scaled_current_frame_off[1] + BlockSize * scale_factor, scaled_current_frame_off[0] + BlockSize * scale_factor - 1),
                                        [0, 0, 255], 1)
    
    previous_It_rect_frame = cv2.rectangle(previous_It_bgr, (scaled_previous_frame_off[1], scaled_previous_frame_off[0]),
                                        (scaled_previous_frame_off[1] + BlockSize * scale_factor, scaled_previous_frame_off[0] + BlockSize * scale_factor - 1),
                                        [0, 255, 0], 1)
       
    
    top_row = np.hstack([previous_rect_frame, current_rect_frame])
    bottom_row = np.hstack([previous_It_rect_frame, current_It_rect_frame])
    splot = np.vstack([top_row, bottom_row])

    last_It = last_It = It.copy()

    # Display the result
    cv2.imshow('Image Grid', splot)
    cv2.waitKey(1)                        
    # DEBUG - end

if __name__ == "__main__":
    main()