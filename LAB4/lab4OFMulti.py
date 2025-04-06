import cv2
import numpy as np
import matplotlib.pyplot as plt



def optical_flow_block_method(I, J, W2=5, dX=5, dY=5):
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
                        distance = np.sqrt(np.sum((JO - IO) ** 2))

                        # Update the best optical flow components if the distance is smaller
                        if distance < min_distance:
                            min_distance = distance
                            best_u, best_v = u, v

            # Assign the best optical flow components to the corresponding pixel
            optical_flow_u[j, i] = best_u
            optical_flow_v[j, i] = best_v

    return optical_flow_u, optical_flow_v

def vis_flow(total_flow_u, total_flow_v, shape, name):
    # Calculate magnitude and angle of the total optical flow
    magnitude, angle = cv2.cartToPolar(total_flow_u, total_flow_v)

    # Convert angle to degrees
    angle_degrees = np.degrees(angle)

    # Resize angle_degrees to match the shape of total_flow_u and total_flow_v
    angle_degrees_resized = cv2.resize(angle_degrees, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize magnitude
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Resize magnitude_normalized to match the shape of the HSV image
    magnitude_resized = cv2.resize(magnitude_normalized, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    # Create HSV image
    hsv = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle_degrees_resized * 0.5  # Scale angle to fit into 0-180 range
    hsv[..., 2] = 255
    hsv[..., 1] = magnitude_resized

    # Convert HSV to RGB for display
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # Display the combined image
    cv2.imshow(name, rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    
def compute_total_optical_flow(pyramid_I_down_scale, pyramid_J_down_scale):
    total_flow_u = np.zeros_like(pyramid_I_down_scale[-1], dtype=np.float32)
    total_flow_v = np.zeros_like(pyramid_I_down_scale[-1], dtype=np.float32)
    
    # Start processing from the smallest scale
    """
    for scale in range(len(pyramid_I_down_scale)-1, -1, -1):
        optical_flow_u, optical_flow_v = optical_flow_block_method(pyramid_I_down_scale[scale], pyramid_J_down_scale[scale])
        vis_flow(optical_flow_u, optical_flow_v, pyramid_I_down_scale[scale].shape, f"Optical Flow Scale {scale}")
    """
    I_new = pyramid_I_down_scale[-1]
    # Iterate over scales except the largest one
    for scale in range(len(pyramid_I_down_scale) - 1, -1, -1):
        # Calculate optical flow
        optical_flow_u, optical_flow_v = optical_flow_block_method(I_new, pyramid_J_down_scale[scale])
        vis_flow(optical_flow_u, optical_flow_v, pyramid_I_down_scale[scale].shape, f"Optical Flow Scale {scale}")
        

        # Modify image according to the flow
        # Your code for modifying the image goes here
        I_new_copy = I_new.copy()
        for i in range(I_new.shape[0]):
            for j in range(I_new.shape[1]):
                if (i + optical_flow_u[i,j] > 0 and i + optical_flow_u[i,j] <I_new.shape[0]) and (j + optical_flow_v[i,j] > 0 and j + optical_flow_v[i,j] <I_new.shape[0]):
                    I_new[int(i+optical_flow_v[i,j]),int(j+optical_flow_u[i,j])] = I_new_copy[i,j]

        # Resize optical flow to match the shape of total_flow_u and total_flow_v
        optical_flow_u_resized = cv2.resize(optical_flow_u*2**scale, (total_flow_u.shape[1], total_flow_u.shape[0]), interpolation=cv2.INTER_LINEAR)
        optical_flow_v_resized = cv2.resize(optical_flow_v*2**scale, (total_flow_v.shape[1], total_flow_v.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Add flows from different scales
        total_flow_u += optical_flow_u_resized
        total_flow_v += optical_flow_v_resized


        # Prepare image for the next larger scale
        if scale != 0:
            I_new = cv2.resize(I_new, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Visualize the total flow
    vis_flow(total_flow_u, total_flow_v, I_new.shape, "Total Optical Flow I and J Images with 5")
def pyramidDownScale(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k-1], (0, 0), fx=0.5, fy=0.5))
    return images



if __name__ == "__main__":
    # Load images
    I = cv2.imread("I.jpg", cv2.IMREAD_GRAYSCALE)
    J = cv2.imread("J.jpg", cv2.IMREAD_GRAYSCALE)
    cm1 = cv2.imread("cm1.png",cv2.IMREAD_GRAYSCALE)
    cm2 = cv2.imread("cm2.png",cv2.IMREAD_GRAYSCALE)
    IJdiff = cv2.absdiff(I,J)

    
    pyramid_I_down_scale = pyramidDownScale(I, 2)
    pyramid_J_down_scale = pyramidDownScale(J, 2)
    pyramid_cm1_down_scale = pyramidDownScale(cm1, 3)
    pyramid_cm2_down_scale = pyramidDownScale(cm2, 3)

     # Initialize total flow matrices
      # Initialize total flow matrices
    compute_total_optical_flow(pyramid_I_down_scale, pyramid_J_down_scale)
    #compute_total_optical_flow(pyramid_cm1_down_scale, pyramid_cm2_down_scale)

   

    

    

    

    
