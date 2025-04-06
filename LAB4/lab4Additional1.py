import cv2
import torch
import numpy as np
import time
#from run_liteflownet import LiteFlowNet
#from run_spynet import SPyNet
print(cv2.__version__)
opt = cv2.optflow.createOptFlow_DualTVL1()


def compute_dense_optical_flow(images, method=cv2.calcOpticalFlowFarneback):
    # Convert images to grayscale
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Initialize empty list to store optical flows
    optical_flows = []

    # Compute optical flow for each consecutive pair of frames
    start_time = time.time()
    for i in range(len(gray_images) - 1):
        # Compute optical flow between current and next frame
        optical_flow = method(gray_images[i], gray_images[i+1], None, 0.5, 3, 15, 10, 5, 1.5, 0)
        optical_flows.append(optical_flow)
    end_time = time.time()

    # Calculate average time per frame
    avg_time_per_frame = (end_time - start_time) / len(gray_images)

    return optical_flows, avg_time_per_frame
def compute_sparse_optical_flow(images, method=cv2.calcOpticalFlowPyrLK, win_size=15):
    # Convert images to grayscale
    gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

    # Initialize points for sparse optical flow
    points = np.array([[x, y] for x in range(0, gray_images[0].shape[1], 10) for y in range(0, gray_images[0].shape[0], 10)], dtype=np.float32)



    # Compute optical flow for each consecutive pair of frames
    start_time = time.time()
    for i in range(len(gray_images) - 1):
        # Compute optical flow between current and next frame
        p0 = points.reshape(-1, 1, 2)
        p1, st, err = method(gray_images[i], gray_images[i+1], p0, None, winSize=(win_size, win_size), maxLevel=2)

        # Visualize optical flow
        img = images[i].copy()

        # Draw circles at detected points
        for jj in range(p0.shape[0]):
            if st[jj] == 1:
                if np.linalg.norm(p1[jj] - p0[jj]) > 2:  # Check if there's a significant change
                    cv2.circle(img, (int(p0[jj, 0, 0]), int(p0[jj, 0, 1])), 3, (0, 0, 255), -1)  # Red circle for detected points
                else:
                    cv2.circle(img, (int(p0[jj, 0, 0]), int(p0[jj, 0, 1])), 2, (0, 255, 0), -1)  # Green circle for unchanged points

        # Display visualization of results
        cv2.imshow("Sparse Optical Flow", img)
        cv2.waitKey(1000)  # Show each frame for 1 second

        # Update points for next iteration based on detected points in the current frame
        points = p0[st == 1].reshape(-1, 1, 2)

        # Append optical flow to the list
        

    end_time = time.time()

    # Calculate average time per frame
    avg_time_per_frame = (end_time - start_time) / len(gray_images)
    print(f"Average time per frame for JK spare optical flow: {avg_time_per_frame} seconds")
"""
def compute_neural_optical_flow(images, network='LiteFlowNet', iStep=1):
    # Load the pre-trained model
    if network == 'LiteFlowNet':
        model = LiteFlowNet()
    elif network == 'SPyNet':
        model = SPyNet()
    else:
        print("Invalid network selected. Choose either 'LiteFlowNet' or 'SPyNet'.")
        return

    # Set the model to evaluation mode
    model.eval()

    # Convert images to tensor
    image1 = torch.from_numpy(images[0]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2 = torch.from_numpy(images[1]).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Pass images through the model
    start_time = time.time()
    with torch.no_grad():
        flow = model(image1, image2)
    end_time = time.time()

    # Calculate average time per frame
    avg_time_per_frame = (end_time - start_time)

    # Convert flow tensor to numpy array
    flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()

    return flow, avg_time_per_frame
    """
def visualize_optical_flow_sequence(optical_flows, shape, name):
    for i, flow in enumerate(optical_flows):
        # Calculate magnitude and angle of the total optical flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Convert angle to degrees
        angle_degrees = np.degrees(angle)

        # Resize angle_degrees to match the shape of flow
        angle_degrees_resized = cv2.resize(angle_degrees, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Normalize magnitude
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Create HSV image
        hsv = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle_degrees_resized * 0.5  # Scale angle to fit into 0-180 range
        hsv[..., 1] = 255
        hsv[..., 2] = magnitude_normalized

        # Convert HSV to RGB for display
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Display the combined image
        cv2.imshow(name, rgb)

        # Wait for a key press before showing the next frame
        cv2.waitKey(0)

if __name__ == "__main__":
    # Load images (replace with your own image sequence)
    with open("pedestrian/temporalROI.txt", "r") as f:
        line = f.readline()
        roi_start, roi_end = map(int, line.split())
    images = [cv2.imread("pedestrian/input/in%06d.jpg" % i) for i in range(roi_start, roi_end)]
    
    # Compute dense optical flow using Farneback's method
    dense_optical_flows, dense_avg_time_per_frame = compute_dense_optical_flow(images)

    # Display dense optical flow results
    visualize_optical_flow_sequence(dense_optical_flows, images[0].shape, "Dense Optical Flow")
    print(f"Average time per frame for dense optical flow: {dense_avg_time_per_frame} seconds")

    # Compute dense optical flow using Lucas-Kanade method and display (PyrLK)
    compute_sparse_optical_flow(images)
    cv2.optflow.createOptFlow_DualTVL1()
    cv2.destroyAllWindows()

    