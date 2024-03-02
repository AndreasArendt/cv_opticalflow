import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def CalcFlow(current_frame, previous_frame):
    # Sobel operator for edge detection in the x-direction and y-direction
    #Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #My = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Scharr
    Mx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    My = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

    # Roberts
    #Mx = np.array([[1, 0], [0, -1]])
    #My = np.array([[0, 1], [-1, 0]])
    Mt = np.array([[-1, -1], [-1, -1]])

    Ix = cv2.filter2D(previous_frame, -1, Mx, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.filter2D(previous_frame, -1, My, borderType=cv2.BORDER_REFLECT)
    It = cv2.filter2D(current_frame, -1, Mt, borderType=cv2.BORDER_REFLECT) + \
         cv2.filter2D(previous_frame, -1, -Mt, borderType=cv2.BORDER_REFLECT)
         
    It = -current_frame + previous_frame

    #cv2.imshow('current_frame', cv2.resize(current_frame, (448, 448), interpolation=cv2.INTER_NEAREST))
    #cv2.imshow('previous_frame', cv2.resize(previous_frame, (448, 448), interpolation=cv2.INTER_NEAREST))
    imgIx = cv2.resize(Ix, (320, 320), interpolation=cv2.INTER_NEAREST)
    imgIy = cv2.resize(Iy, (320, 320), interpolation=cv2.INTER_NEAREST)
    imgIt = cv2.resize(It, (320, 320), interpolation=cv2.INTER_NEAREST)    
    
    splot = np.hstack([imgIx, imgIy, imgIt])
    cv2.imshow('Ix, Iy, It', splot)
    cv2.waitKey(1)

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
        plt.figure(4)
        plt.cla()
        plt.imshow(previous_frame, cmap='gray')        
        u_norm = u / (np.sqrt( u**2 + v**2 ))
        v_norm = v / (np.sqrt( u**2 + v**2 ))
        plt.quiver(previous_frame.shape[1]//2, previous_frame.shape[0]//2, v_norm, u_norm, color='red', scale_units='width', scale=10)
        plt.pause(1e-3)

    return u, v, flow_u, flow_v, Ix, Iy, It