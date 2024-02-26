import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def CalcFlow(current_frame, previous_frame):
    # Sobel operator for edge detection in the x-direction and y-direction
    #Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    #My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).T

    Mx = np.array([[-1, -1], [1, 1]]).T
    My = np.array([[-1, -1], [1, 1]])
    Mt = np.array([[-1, -1], [-1, -1]])

    #Mx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    #My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Combined filter for temporal gradient
    Mt = np.array([1, 1]) - np.array([-1, -1])
    Mt = Mt.reshape((-1, 1))  # Reshape to a column vector

    Ix = cv2.filter2D(previous_frame, -1, Mx, borderType=cv2.BORDER_REFLECT) + cv2.filter2D(current_frame, -1, Mx, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.filter2D(previous_frame, -1, My, borderType=cv2.BORDER_REFLECT) + cv2.filter2D(current_frame, -1, My, borderType=cv2.BORDER_REFLECT)
    It = cv2.filter2D(previous_frame, -1, Mt, borderType=cv2.BORDER_REFLECT) + cv2.filter2D(current_frame, -1, -Mt, borderType=cv2.BORDER_REFLECT)

    # cv2.imshow('current_frame', cv2.resize(current_frame, (448, 448), interpolation=cv2.INTER_NEAREST))
    # cv2.imshow('previous_frame', cv2.resize(previous_frame, (448, 448), interpolation=cv2.INTER_NEAREST))
    # cv2.imshow('Ix', cv2.resize(Ix, (448, 448), interpolation=cv2.INTER_NEAREST))
    # cv2.imshow('Iy', cv2.resize(Iy, (448, 448), interpolation=cv2.INTER_NEAREST))
    # cv2.imshow('It', cv2.resize(It, (448, 448), interpolation=cv2.INTER_NEAREST))
    # cv2.imshow('Itt', cv2.resize(Itt, (448, 448), interpolation=cv2.INTER_NEAREST))
    #cv2.waitKey(1)

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