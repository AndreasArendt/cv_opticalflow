import cv2
import matplotlib.pyplot as plt
import numpy as np


def CalcFlow(current_frame, previous_frame):
    # Sobel operator for edge detection in the x-direction and y-direction
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).T

    # Combined filter for temporal gradient
    Mt = np.array([1, 1]) - np.array([-1, -1])
    Mt = Mt.reshape((-1, 1))  # Reshape to a column vector

    Ix = cv2.filter2D(previous_frame, -1, Mx)
    Iy = cv2.filter2D(previous_frame, -1, My)
    It = cv2.filter2D(previous_frame, -1, Mt) + cv2.filter2D(current_frame, -1, -Mt)

    cv2.imshow('Ix', cv2.resize(Ix, (448, 448), interpolation=cv2.INTER_NEAREST))
    cv2.imshow('Iy', cv2.resize(Iy, (448, 448), interpolation=cv2.INTER_NEAREST))
    cv2.imshow('It', cv2.resize(It, (448, 448), interpolation=cv2.INTER_NEAREST))
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
        plt.figure(1)
        plt.cla()
        plt.imshow(previous_frame, cmap='gray')
        plt.quiver(np.arange(0, previous_frame.shape[1]), np.arange(0, previous_frame.shape[0]), u, v, color='red', scale=100)
        plt.pause(1e-3)

    elevation = 1 / np.sqrt(u * u + v * v)
    azimuth = np.sign(v) * u / np.sqrt(u * u + v * v)

    #u = u - (2*MaxMovement + BlockSize)
    #v = v - (2*MaxMovement + BlockSize)

    return u, v, flow_u, flow_v, Ix, Iy, It, elevation, azimuth