import cv2
import numpy as np
from statistics import mean, stdev

def load_images(directory, start_frame, end_frame):
    images = []
    for i in range(start_frame, end_frame):
        img = cv2.imread(directory + "/in%06d.jpg" % i)
        if img is not None:
            images.append(img)
    return images

def convert_to_grayscale(images):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

def apply_background_subtraction(images):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)
    fg_masks = []
    for img in images:
        fg_mask = bg_subtractor.apply(img)
        _, fg_mask= cv2.threshold(fg_mask, 40, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_masks.append(fg_mask)
    return fg_masks

def compute_dense_optical_flow(images):
    optical_flows = []
    for i in range(len(images) - 1):
        flow = cv2.calcOpticalFlowFarneback(images[i], images[i+1], None, 0.5, 3, 15, 10, 5, 1.5, 0)
        optical_flows.append(flow)
    return optical_flows

def label_objects(fg_masks):
    labeled_objects = [cv2.connectedComponents(fg_mask)[1] for fg_mask in fg_masks]
    return labeled_objects

def analyze_optical_flow(labeled_objects, optical_flows):
    modulus_means_list = []
    modulus_stdevs_list = []
    angle_means_list = []
    angle_stdevs_list = []

    for objects, flow in zip(labeled_objects, optical_flows):
        modulus_means = []
        modulus_stdevs = []
        angle_means = []
        angle_stdevs = []

        # Iterate over the objects detected in the current frame
        for label in range(1, np.max(objects) + 1):
            mask = (objects == label).astype(np.uint8)

            # Compute the modulus and angle of the optical flow vectors within the mask
            modulus = np.linalg.norm(flow, axis=-1)
            angle = np.arctan2(flow[..., 1], flow[..., 0])

            # Compute mean and standard deviation for modulus and angle within the mask
            mean_modulus = np.mean(modulus[mask > 0])
            std_modulus = np.std(modulus[mask > 0])
            mean_angle = np.mean(angle[mask > 0])
            std_angle = np.std(angle[mask > 0])

            # Append the statistics to the lists
            modulus_means.append(mean_modulus)
            modulus_stdevs.append(std_modulus)
            angle_means.append(mean_angle)
            angle_stdevs.append(std_angle)

        # Append the lists for the current frame to the main lists
        modulus_means_list.append(modulus_means)
        modulus_stdevs_list.append(modulus_stdevs)
        angle_means_list.append(angle_means)
        angle_stdevs_list.append(angle_stdevs)

    return modulus_means_list, modulus_stdevs_list, angle_means_list, angle_stdevs_list
def visualize_results(images, labeled_objects, optical_flows, fg_masks, modulus_means_list, modulus_stdevs_list, angle_means_list, angle_stdevs_list, min_area_threshold=150):
    for img, objects, flow, fg_mask, modulus_means, modulus_stdevs, angle_means, angle_stdevs in zip(images, labeled_objects, optical_flows, fg_masks, modulus_means_list, modulus_stdevs_list, angle_means_list, angle_stdevs_list):
        # Create a blank canvas to display multiple images
        canvas = np.zeros((2 * img.shape[0], 4 * img.shape[1], 3), dtype=np.uint8)

        # Display the Farneback optical flow image (top left)
        optical_flow_img = np.zeros_like(img)
        optical_flow_img[..., 0] = flow[..., 0]
        optical_flow_img[..., 1] = flow[..., 1]
        optical_flow_img = cv2.normalize(optical_flow_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = optical_flow_img

        # Display the foreground object segmentation image (top right)
        fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        canvas[:img.shape[0], img.shape[1]:2 * img.shape[1]] = fg_mask_bgr

        # Display the labeling image (bottom left)
        labeled_img = np.zeros_like(img)
        for label in range(1, np.max(objects) + 1):
            mask = (objects == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = (60, 60, 60)  
            cv2.drawContours(labeled_img, contours, -1, color, 2)
        canvas[img.shape[0]:, :img.shape[1]] = labeled_img

        # Iterate over the objects detected in the current frame and draw rectangles with parameters (bottom right)
        for label, (modulus_mean, modulus_stdev, angle_mean, angle_stdev) in enumerate(zip(modulus_means, modulus_stdevs, angle_means, angle_stdevs), start=1):
            # Generate mask for the current object label
            mask = np.zeros_like(objects, dtype=np.uint8)
            mask[objects == label] = 255
            
            # Find the bounding box of the object
            x, y, w, h = cv2.boundingRect(mask)
            
            # Compute the area of the bounding rectangle
            area = w * h
            
            # Draw rectangle around the object if area is above threshold
            if area >= min_area_threshold:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw mean and standard deviation text inside the rectangle
                text = "MeanMo:{:.2f},StdMo:{:.2f}".format(modulus_mean, modulus_stdev)
                cv2.putText(img, text, (x + 5, y + h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                text = "MeanA:{:.2f}, StdA:{:.2f}".format(angle_mean, angle_stdev)
                cv2.putText(img, text, (x + 5, y + h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Display the result image for the current frame (bottom right)
        canvas[img.shape[0]:,  2*img.shape[1]:3 * img.shape[1]] = img

        # Show the canvas with all images
        cv2.imshow("Result", canvas)
        cv2.waitKey(0)  # Adjust delay as needed
    
    cv2.destroyAllWindows()





if __name__ == "__main__":
    directory = "highway/input"
    with open("highway/temporalROI.txt", "r") as f:
        line = f.readline()
        start_frame, end_frame = map(int, line.split())

    images = load_images(directory, start_frame, end_frame)
    gray_images = convert_to_grayscale(images)
    fg_masks = apply_background_subtraction(gray_images)
    optical_flows = compute_dense_optical_flow(fg_masks)
    labeled_objects = label_objects(fg_masks)
    modulus_means, modulus_stdevs, angle_means, angle_stdevs = analyze_optical_flow(labeled_objects, optical_flows)

    visualize_results(images, labeled_objects, optical_flows,fg_masks,modulus_means,modulus_stdevs,angle_means,angle_stdevs)

   
