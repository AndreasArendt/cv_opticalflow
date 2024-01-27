import cv2
import numpy as np
import matplotlib.pyplot as plt

BlockSize   = 80 # Block Size of chunks (must be gcd of width/height)
MaxMovement = 20 # maximum movement of rectangle from starting point

WIDTH = 1280
HEIGHT = 720

def main():

    # Capture video from file or camera
    # cap = cv2.VideoCapture('slow_traffic_small.mp4')
    # cap = cv2.VideoCapture('uav_1.mp4')
    cap = cv2.VideoCapture('data/uav_2.mp4')

    #skip first frames
    for i in range(1,30):
        ret, frame = cap.read()       

    # find blocks from prev_frame in current_frame
    ret, frame = cap.read()       
    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fcnt = 0

    while True:
        ret, frame = cap.read()       
        fcnt += 1

        if not ret:
            break
        
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        f_prev = np.array(prev_frame)

        best_ssd = -np.Inf
        current_frame_off = np.Inf, np.Inf
        previous_frame_off = np.Inf, np.Inf

        # iterate over each column, row in previous frame
        for c in range(0, WIDTH, BlockSize):
            for r in range(0, HEIGHT, BlockSize):        
                prev_block = f_prev[r:r+BlockSize, c:c+BlockSize]
                
                # limit row and column indices
                c_idx_st = np.max([c-MaxMovement, 0])
                c_idx_en = np.min([c+MaxMovement+BlockSize, WIDTH])

                r_idx_st = np.max([r-MaxMovement, 0])
                r_idx_en = np.min([r+MaxMovement+BlockSize, HEIGHT])

                macro_block = current_frame[r_idx_st:r_idx_en, c_idx_st:c_idx_en]

                # run block matching
                ssd, c_off, r_off = BlockMatching(macro_block, prev_block)

                if ssd > best_ssd:
                    best_ssd = ssd
                    current_frame_off = r,c
                    previous_frame_off = r_off+r,c_off+c
                                        
                # rect_frame = cv2.rectangle(prev_frame.copy(), (c, r) , (c+BlockSize, r+BlockSize), [0,0,255], 1)           
                # cv2.imshow("prev_frame_rect", rect_frame)
                # cv2.imshow("prev_block", prev_block)
                # cv2.waitKey(1)
        
        rect_frame = cv2.rectangle(frame.copy(), (current_frame_off[1], current_frame_off[0]) , (current_frame_off[1]+BlockSize, current_frame_off[0]+BlockSize-1), [0,0,255], 1)           
        rect_frame = cv2.rectangle(rect_frame, (previous_frame_off[1], previous_frame_off[0] ), (previous_frame_off[1]+BlockSize, previous_frame_off[0]+BlockSize-1), [0,255,0], 1)           

        current_subframe  = current_frame[current_frame_off[0]:current_frame_off[0]+BlockSize, current_frame_off[1]:current_frame_off[1]+BlockSize]
        previous_subframe = prev_frame[previous_frame_off[0]:previous_frame_off[0]+BlockSize, previous_frame_off[1]:previous_frame_off[1]+BlockSize]

        cv2.imshow("current_subframe", current_subframe)        
        cv2.imshow("previous_subframe", previous_subframe)
        cv2.imshow("prev_frame ssd", rect_frame)        
        cv2.waitKey(1)

        # Calc Optical Flow
        u, v, flow_u, flow_v = CalcFlow(current_subframe, previous_subframe)

        plt.subplot(1,2,1)
        plt.plot(fcnt, u, marker='o', color='blue')

        plt.subplot(1,2,2)
        plt.plot(fcnt, v, marker='o', color='blue')
        plt.pause(1e-10)

        prev_frame = current_frame

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            plt.show()
            break

    # Turn off interactive mode and close the OpenCV windows
    cv2.destroyAllWindows()
    cap.release()


def BlockMatching(macro_block, previous_frame):
    # offset - shifting
    max_c = macro_block.shape[1] - previous_frame.shape[1]
    max_r = macro_block.shape[0] - previous_frame.shape[0]

    best_ssd = np.inf

    STEPSIZE = 5 # parameter to play around - improves performance, with precision loss

    for c_off in range(0,max_c,STEPSIZE):
        for r_off in range(0,max_r,STEPSIZE):
            m = macro_block[r_off:r_off+BlockSize, c_off:c_off+BlockSize]
            p = previous_frame

            # Calculate Sum of Squared Differences (SSD) metric
            ssd = np.sum((m - p) ** 2)

            # Update best match if the current SSD is smaller
            if ssd < best_ssd:
                best_ssd = ssd
                best_match_c = np.max([c_off - MaxMovement, 0])
                best_match_r = np.max([r_off - MaxMovement, 0])
            
            #rect_frame = cv2.rectangle(macro_block.copy(), (c_off, r_off) , (c_off+BlockSize-1, r_off+BlockSize-1), [0,0,255], 3)                                   
            #cv2.imshow("macro_block", rect_frame)
            #cv2.imshow("curr + prev + delta", cv2.hconcat([m, p, m-p]))
            #cv2.waitKey(1)
            
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