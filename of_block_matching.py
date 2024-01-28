import cv2
import numpy as np
import matplotlib.pyplot as plt

BlockSize   = 16 #80 # Block Size of chunks (must be gcd of width/height)
MaxMovement = 32 # maximum movement of rectangle from starting point

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

    HEIGHT = frame.shape[0]
    WIDTH = frame.shape[1]

    fcnt = 0

    u_arr = np.array([])
    v_arr = np.array([])

    #while True:
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
        for c in range(MaxMovement//2, WIDTH-MaxMovement//2, BlockSize):
            for r in range(MaxMovement//2, HEIGHT-MaxMovement//2, BlockSize):        
                # limit row and column indices
                c_idx_st = np.max([c-MaxMovement, 0])
                c_idx_en = np.min([c+MaxMovement+BlockSize, WIDTH])

                r_idx_st = np.max([r-MaxMovement, 0])
                r_idx_en = np.min([r+MaxMovement+BlockSize, HEIGHT])

                macro_block = current_frame[r_idx_st:r_idx_en, c_idx_st:c_idx_en]
                prev_block = prev_frame[r:r+BlockSize, c:c+BlockSize]

                # run block matching
                ssd, c_off, r_off = BlockMatching(macro_block, prev_block)

                if ssd < best_ssd:
                    best_ssd = ssd
                    current_frame_off = r,c
                    previous_frame_off = r_off+r_idx_st, c_off+c_idx_st
                                        
                # rect_frame = cv2.rectangle(prev_frame.copy(), (c, r) , (c+BlockSize, r+BlockSize), [0,0,255], 1)           
                # cv2.imshow("prev_frame_rect", rect_frame)
                # cv2.imshow("prev_block", prev_block)
                # cv2.waitKey(1)

        current_subframe  = current_frame[current_frame_off[0]:current_frame_off[0]+BlockSize, current_frame_off[1]:current_frame_off[1]+BlockSize]
        previous_subframe = prev_frame[previous_frame_off[0]:previous_frame_off[0]+BlockSize, previous_frame_off[1]:previous_frame_off[1]+BlockSize]
                
        rect_frame = cv2.rectangle(frame.copy(), (current_frame_off[1], current_frame_off[0]) , (current_frame_off[1]+BlockSize, current_frame_off[0]+BlockSize-1), [0,0,255], 1)           
        rect_frame = cv2.rectangle(rect_frame, (previous_frame_off[1], previous_frame_off[0] ), (previous_frame_off[1]+BlockSize, previous_frame_off[0]+BlockSize-1), [0,255,0], 1)           
        
        # Increase the size of the plot
        large_current_subframe = cv2.resize(current_subframe.copy(), (480, 480), interpolation=cv2.INTER_NEAREST)
        large_previous_subframe = cv2.resize(previous_subframe.copy(), (480, 480), interpolation=cv2.INTER_NEAREST)
        rect_frame = cv2.resize(rect_frame, (480, 480), interpolation=cv2.INTER_NEAREST)
        
        #cv2.imshow("current_subframe", large_current_subframe)        
        #cv2.imshow("previous_subframe", large_previous_subframe)

        prev_rect = cv2.rectangle(prev_frame.copy(), (previous_frame_off[1], previous_frame_off[0] ), (previous_frame_off[1]+BlockSize, previous_frame_off[0]+BlockSize-1), [0,255,0], 1)           
        prev_rect = cv2.resize(prev_rect, (480, 480), interpolation=cv2.INTER_NEAREST)

        #cv2.imshow("prev_frame ssd", prev_rect)        
        #cv2.imshow("rect_Frame ssd", rect_frame)       

        # prev_rect_col = cv2.cvtColor(prev_rect, cv2.COLOR_BAYER_BG2BGR)
        # large_previous_subframe = cv2.cvtColor(large_previous_subframe, cv2.COLOR_BAYER_BG2BGR)
        # large_current_subframe = cv2.cvtColor(large_current_subframe, cv2.COLOR_BAYER_BG2BGR)

        # top_row = cv2.hconcat([large_previous_subframe, large_current_subframe])
        # bottom_row = cv2.hconcat([prev_rect_col, rect_frame])
        # grid = cv2.vconcat([top_row, bottom_row])

        # # Display the grid
        # cv2.imshow("Image Grid", grid) 
        # cv2.waitKey(1)

        # Calc Optical Flow
        u, v, flow_u, flow_v = CalcFlow(current_subframe, previous_subframe)        
        
        u_arr = np.append(u_arr,u)
        v_arr = np.append(v_arr,v)

        # plt.subplot(1,3,1)
        # plt.plot(fcnt, u, marker='o', color='blue')

        # plt.subplot(1,3,2)
        # plt.plot(fcnt, v, marker='o', color='blue')

        # plt.subplot(1,3,3)
        # plt.plot(fcnt, ang, marker='o', color='blue')

        # plt.pause(1e-10)

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



def BlockMatching(macro_block, previous_frame):
    # offset - shifting
    max_c = macro_block.shape[1] - previous_frame.shape[1]
    max_r = macro_block.shape[0] - previous_frame.shape[0]

    best_ssd = np.inf

    STEPSIZE = 1 # parameter to play around - improves performance, with precision loss

    for c_off in range(0,max_c,STEPSIZE):
        for r_off in range(0,max_r,STEPSIZE):
            m = macro_block[r_off:r_off+BlockSize, c_off:c_off+BlockSize]
            p = previous_frame

            # Calculate Sum of Squared Differences (SSD) metric
            ssd = np.sum((m - p) ** 2)

            # Update best match if the current SSD is smaller
            if ssd < best_ssd:
                best_ssd = ssd
                best_match_c = c_off
                best_match_r = r_off
            
            rect_frame = cv2.rectangle(macro_block.copy(), (c_off, r_off) , (c_off+BlockSize-1, r_off+BlockSize-1), [0,0,255], 1)                                   
            rect_frame_resize = cv2.resize(rect_frame, (480, 480), interpolation=cv2.INTER_NEAREST)
            mpd = cv2.hconcat([m, p, m-p])
            mpd_resize = cv2.resize(mpd, (480, 480), interpolation=cv2.INTER_NEAREST)

            # cv2.imshow("macro_block", rect_frame_resize)
            # cv2.imshow("curr + prev + delta", mpd_resize)
            # cv2.waitKey(1)
            
    #print("Best ssd: " + str(best_ssd))
    #print("at: " + str(c_off) +  ", " + str(r_off))

    return best_ssd, best_match_c, best_match_r

def CalcFlow(current_frame, previous_frame):
    # Sobel operator for edge detection in the x-direction and y-direction
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Combined filter for temporal gradient
    Mt = np.array([1, 1, 1]) - np.array([-1, -1, -1])

    Ix = cv2.filter2D(previous_frame, -1, Mx)
    Iy = cv2.filter2D(previous_frame, -1, My)
    It = cv2.filter2D(previous_frame, -1, Mt) + cv2.filter2D(current_frame, -1, -Mt)

#    cv2.imshow("Ix", cv2.hconcat(Ix, Iy, It))

    # Subsample the features based on the grid
    fx = Ix.flatten()
    fy = Iy.flatten()
    ft = It.flatten()

    A = np.vstack((fx, fy)).T
    y_subsampled = -ft

    u, v = np.linalg.lstsq(A, y_subsampled, rcond=None)[0]

    # Optionally, visualize the full-resolution optical flow
    flow_u = u * Ix
    flow_v = v * Iy

    return u, v, flow_u, flow_v

if __name__ == "__main__":
    main()