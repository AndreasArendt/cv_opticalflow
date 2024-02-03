import cv2
import numpy as np
import matplotlib.pyplot as plt

BlockSize   = 16 #80 # Block Size of chunks (must be gcd of width/height)
MaxMovement = 64-16 # maximum movement of rectangle from starting point

def main():

    # Capture video from file or camera
    # cap = cv2.VideoCapture('slow_traffic_small.mp4')
    # cap = cv2.VideoCapture('uav_1.mp4')
    # cap = cv2.VideoCapture('data/uav_2.mp4')
    cap = cv2.VideoCapture('data/vid.mp4')

    #skip first frames
    for i in range(1,1599):#+75):
        ret, frame = cap.read()       

    # find blocks from prev_frame in current_frame
    ret, frame = cap.read()       
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    fcnt = 0

    u_arr = np.array([])
    v_arr = np.array([])


    # frame = cv2.imread('data/clinic/img1.png')
    # prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    HEIGHT = frame.shape[0]
    WIDTH = frame.shape[1]

    #while True:
    for ii in range(1,700):# range(1,700):
        ret, frame = cap.read()       

        #frame = cv2.imread('data/clinic/img' + str(ii) + '.png')
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Processing frame: " + str(ii))
        
        fcnt += 1

        if not ret:
            break        

        best_ssd = np.Inf
        current_frame_off = np.Inf, np.Inf
        previous_frame_off = np.Inf, np.Inf

        
        # row,col,ch= frame.shape
        # mean = 0
        # var = 0.5
        # sigma = var**0.5
        # gauss = np.random.normal(mean,sigma,(row,col,ch))
        # gauss = gauss.reshape(row,col,ch)
        # gauss = gauss.astype(np.uint8)
        # current_frame = cv2.cvtColor(frame + gauss, cv2.COLOR_BGR2GRAY)

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
                ssd, c_off, r_off = BlockMatching(macro_block, prev_block)
                
                if ssd < best_ssd and ssd > 0:
                    best_ssd = ssd
                    current_frame_off = r_off+r_idx_st, c_off+c_idx_st
                    previous_frame_off = r,c
                 
        
        print(str(r_off) + ", " + str(c_off))

        current_subframe  = current_frame[current_frame_off[0]:current_frame_off[0]+BlockSize, current_frame_off[1]:current_frame_off[1]+BlockSize]
        previous_subframe = prev_frame[previous_frame_off[0]:previous_frame_off[0]+BlockSize, previous_frame_off[1]:previous_frame_off[1]+BlockSize]
                
        # Calc Optical Flow
        u, v, flow_u, flow_v, Ix, Iy, It = CalcFlow(current_subframe, previous_subframe)        
        #_u, _v, _flow_u, _flow_v, _Ix, _Iy, _It = CalcFlow(current_frame, prev_frame)        
        
        debug = True
        if debug == True:
            DebugPlot(current_frame, current_frame_off, prev_frame, previous_frame_off, Ix, Iy ,It)
            print(str(u) + ", " + str(v), ", ssd: " + str(ssd))

        u_arr = np.append(u_arr,u)
        v_arr = np.append(v_arr,v)
       
        prev_frame = current_frame

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            plt.show()
            break

    # Turn off interactive mode and close the OpenCV windows
    cv2.destroyAllWindows()
    cap.release()

    combined_array = np.column_stack((u_arr, v_arr))

    # Save the combined array to a CSV file
    np.savetxt('./out.csv', combined_array, delimiter=',', header='u_arr,v_arr', comments='')

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

def BlockMatching(macro_block, prev_block):
    # offset - shifting
    max_c = macro_block.shape[1] - prev_block.shape[1]
    max_r = macro_block.shape[0] - prev_block.shape[0]

    best_ssd = np.inf
    best_match_c = np.inf
    best_match_r = np.inf

    STEPSIZE = 1 # parameter to play around - improves performance, with precision loss

    for c_off in range(0,max_c+1, STEPSIZE):
        for r_off in range(0,max_r+1,STEPSIZE):
            m = macro_block[r_off:r_off+BlockSize, c_off:c_off+BlockSize]

            # Calculate Sum of Squared Differences (SSD) metric
            ssd = np.sum((m - prev_block) ** 2)

            # Update best match if the current SSD is smaller
            if ssd < best_ssd and ssd > 0:
                best_ssd = ssd
                best_match_c = c_off
                best_match_r = r_off

            debug = False
            if debug == True:                      
                large_macro_block = cv2.resize(macro_block, (448, 448), interpolation=cv2.INTER_NEAREST)                
                large_prev_block = cv2.resize(prev_block, (448, 448), interpolation=cv2.INTER_NEAREST)                

                macro_block_bgr = cv2.cvtColor(large_macro_block, cv2.COLOR_BAYER_BG2BGR)
                prev_block_bgr = cv2.cvtColor(large_prev_block, cv2.COLOR_BAYER_BG2BGR)

                row_scale_factor = 448//macro_block.shape[0]
                col_scale_factor = 448//macro_block.shape[1]
                scaled_frame = (r_off * row_scale_factor, c_off * col_scale_factor)

                # Draw rectangles on the frames with scaled coordinates
                rect = cv2.rectangle(macro_block_bgr.copy(), (scaled_frame[1], scaled_frame[0]),
                                        (scaled_frame[1] + BlockSize * col_scale_factor -1, scaled_frame[0] + BlockSize * row_scale_factor -1),
                                        [0, 0, 255], 1)
                rect = cv2.rectangle(rect, (0,0,), (448 -2, 448 -2),
                                        [255, 0, 255], 1)

                splot = np.hstack([rect, prev_block_bgr])
                cv2.imshow('Block Matchin - debug', splot)
                cv2.waitKey(1)  
                
    return best_ssd, best_match_c, best_match_r

def CalcFlow(current_frame, previous_frame):
    # Sobel operator for edge detection in the x-direction and y-direction
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Combined filter for temporal gradient
    Mt = np.array([1, 1]) - np.array([-1, -1])
    Mt = Mt.reshape((-1, 1))  # Reshape to a column vector

    Ix = cv2.filter2D(previous_frame, -1, Mx)
    Iy = cv2.filter2D(previous_frame, -1, My)
    It = cv2.filter2D(previous_frame, -1, Mt) + cv2.filter2D(current_frame, -1, -Mt)
    
    fx = Ix.flatten()
    fy = Iy.flatten()
    ft = It.flatten()

    A = np.vstack((fx, fy)).T
    y = -ft

    u, v = np.linalg.lstsq(A, y, rcond=None)[0]

    # Optionally, visualize the full-resolution optical flow
    flow_u = u * Ix
    flow_v = v * Iy

    debug = False
    if debug == True:
        plt.figure(1)
        plt.cla()
        plt.imshow(previous_frame, cmap='gray')
        plt.quiver(np.arange(0, previous_frame.shape[1]), np.arange(0, previous_frame.shape[0]), u, v, color='red', scale=100)
        plt.pause(1e-3)

    return u, v, flow_u, flow_v, Ix, Iy, It

if __name__ == "__main__":
    main()