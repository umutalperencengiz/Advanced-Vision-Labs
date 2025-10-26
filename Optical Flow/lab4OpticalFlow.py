import cv2
import numpy as np
import matplotlib.pyplot as plt

def optical_flow_block_method(I, J, W2=3, dX=3, dY=3):
    # Initialize optical flow matrices
    optical_flow_u = np.zeros_like(I, dtype=np.float32)
    optical_flow_v = np.zeros_like(I, dtype=np.float32)

    # Iterate over the image, excluding edges
    for j in range(W2, I.shape[0] - W2):
        for i in range(W2, I.shape[1] - W2):
            # Cut out a part of the I frame
            IO = np.float32(I[j-W2:j+W2+1, i-W2:i+W2+1])

            # Search for motion around the pixel (j, i)
            min_distance = float('inf')
            best_u = best_v = 0

            # Iterate over the search window
            for v in range(-dY, dY+1):
                for u in range(-dX, dX+1):
                    # Calculate the indices for the pixel in J
                    jv = j + v
                    iu = i + u

                    # Check if the indices are within bounds
                    if (0 <= jv - W2 < J.shape[0] and 0 <= jv + W2 + 1 < J.shape[0] and
                            0 <= iu - W2 < J.shape[1] and 0 <= iu + W2 + 1 < J.shape[1]):
                        # Cut out the surrounding of pixel (j+v, i+u) from J
                        JO = np.float32(J[jv-W2:jv+W2+1, iu-W2:iu+W2+1])

                        # Calculate the Euclidean distance between IO and JO
                        distance = np.sqrt(np.sum((np.square(JO-IO))))

                        # Update the best optical flow components if the distance is smaller
                        if distance < min_distance:
                            min_distance = distance
                            best_u, best_v = u, v

            # Assign the best optical flow components to the corresponding pixel
            optical_flow_u[j, i] = best_u
            optical_flow_v[j, i] = best_v

    return optical_flow_u, optical_flow_v

if __name__ == "__main__":
    # Load images
    I = cv2.imread("I.jpg", cv2.IMREAD_GRAYSCALE)
    J = cv2.imread("J.jpg", cv2.IMREAD_GRAYSCALE)
    IJdiff = cv2.absdiff(I,J)

    # Apply block method for optical flow estimation
    optical_flow_u, optical_flow_v = optical_flow_block_method(I, J)

    # Convert optical flow to polar coordinates
    magnitude, angle = cv2.cartToPolar(optical_flow_u, optical_flow_v)

    # Normalize magnitude
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert angle to degrees
    angle_degrees = np.degrees(angle)

    # Convert angle range to 0-180
    angle_degrees_0_180 = angle_degrees * 90 / np.pi

    # Create HSV image
    hsv = np.zeros((I.shape[0], I.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle_degrees_0_180
    hsv[..., 2] = 255
    hsv[..., 1] = magnitude_normalized

    # Convert HSV to BGR for display
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Display optical flow
    cv2.imshow("Normal Image",I)
    cv2.imshow('Optical Flow', bgr)
    cv2.imshow("IJdiff",IJdiff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
