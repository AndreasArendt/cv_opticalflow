import cv2
import numpy as np
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture('slow_traffic_small.mp4')
# cap = cv2.VideoCapture('uav_1.mp4')
cap = cv2.VideoCapture('uav_2.mp4')

ret, frame1 = cap.read()
gframe_1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gframe_2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Sobel operator for edge detection in the x-direction and y-direction
    Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Combined filter for temporal gradient
    Mt = np.array([1, 1, 1]) - np.array([-1, -1, -1])

    Ix = cv2.filter2D(gframe_1, -1, Mx)
    Iy = cv2.filter2D(gframe_1, -1, My)
    It = cv2.filter2D(gframe_1, -1, Mt) + cv2.filter2D(gframe_2, -1, -Mt)

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
    
    # Display the original and processed frames
    cv2.imshow("Original Frame", gframe_1)
    cv2.imshow("Optical Flow Visualization", flow_v)


    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

    # Update for the next iteration
    gframe_1 = gframe_2

# Turn off interactive mode and close the OpenCV windows
cv2.destroyAllWindows()
cap.release()
