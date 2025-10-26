import cv2
import numpy as np
import os
import time
import scipy
from joblib import Parallel, delayed


def load_images(image_folder, start_idx, end_idx):
    images = []
    for idx in range(start_idx, end_idx):
        image = cv2.imread(os.path.join(image_folder, "in{:06d}.jpg".format(idx)), cv2.IMREAD_GRAYSCALE)
        images.append(image)
    return images
def load_groundtruth_images(image_folder, start_idx, end_idx):
    images = []
    for idx in range(start_idx, end_idx):
        image = cv2.imread(os.path.join(image_folder, "gt{:06d}.png".format(idx)), cv2.IMREAD_GRAYSCALE)
        images.append(image)
    return images
def compute_metrics(object_mask, reference_mask):
    # Resize reference mask to match object mask
    reference_mask_resized = cv2.resize(reference_mask, (object_mask.shape[1], object_mask.shape[0]))

    # Compute true positives (TP), false positives (FP), false negatives (FN)
    TP_M = np.logical_and((object_mask == 255), (reference_mask_resized == 255))
    FP_M = np.logical_and((object_mask == 255), (reference_mask_resized == 0))
    FN_M = np.logical_and((object_mask == 0), (reference_mask_resized == 255))

    TP = np.sum(TP_M)
    FP = np.sum(FP_M)
    FN = np.sum(FN_M)
    return TP, FP, FN

def vibe_background_subtraction(image_folder, groundtruth_folder, roi_start, roi_end, N=20, R=20, _min=2, update_factor=16, resize_factor=0.5):
    TP_total, FP_total, FN_total = 0, 0, 0
    # Load images
    start_time = time.time()
    images = load_images(image_folder, roi_start, roi_end)
    groundtruth_images = load_groundtruth_images(groundtruth_folder, roi_start, roi_end)
    print("Time taken to load images:", time.time() - start_time)
    resized_images = [cv2.resize(img, None, fx=resize_factor, fy=resize_factor) for img in images]
    samples = np.array(resized_images[:N])

    # Perform background subtraction
    for idx, (gray_frame, reference_mask) in enumerate(zip(resized_images[N:], groundtruth_images[N:]), start=roi_start+N):

        # Compute foreground mask
        foreground_mask = np.zeros_like(gray_frame, dtype=np.uint8)
        for i in range(gray_frame.shape[0]):
            for j in range(gray_frame.shape[1]):
                count, index, dist = 0, 0, 0
                while count < _min and index < N:
                    try:
                        dist = np.abs(np.int16(gray_frame[i, j]) - np.int16(samples[index, i, j]))
                    except OverflowError:
                        dist = 255
                    if dist < R:
                        count += 1
                    index += 1
                if count >= _min:
                    foreground_mask[i, j] = 0  # Background
                else:
                    foreground_mask[i, j] = 255  # Foreground
        
        # Apply temporal filtering (median filter)
        foreground_mask = cv2.medianBlur(foreground_mask, 5)
        
        # Randomization: Update background model randomly
        for _ in range(int(gray_frame.size * update_factor / 100)):
            x, y = np.random.randint(0, gray_frame.shape[0]), np.random.randint(0, gray_frame.shape[1])
            samples[np.random.randint(0, N), x, y] = gray_frame[x, y]

        TP, FP, FN = compute_metrics(foreground_mask, reference_mask)
        TP_total += TP
        FP_total += FP
        FN_total += FN

        cv2.imshow('Foreground Mask', foreground_mask)
        cv2.imshow('Original Frame', images[idx - roi_start])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

    # Calculate precision, recall, F1 score
    P_total = TP_total / (TP_total + FP_total) if TP_total + FP_total > 0 else 0
    R_total = TP_total / (TP_total + FN_total) if TP_total + FN_total > 0 else 0
    F1_total = 2 * P_total * R_total / (P_total + R_total) if P_total + R_total > 0 else 0
    print("Overall Precision:", P_total)
    print("Overall Recall:", R_total)
    print("Overall F1 Score:", F1_total)

def pbas_background_subtraction(input_dir, gt_dir, roi_start, roi_end, buffer_size=30, step=5, N=20, min_matches=2, phi=10, alpha=0.05, R_lower=18, R_upper=255):
    # Function to perform the frequency analysis to determine the background model
    def mode_frequency_analysis(buffer):
        buffer = np.dstack(buffer)
        pixel_frequencies = scipy.stats.mode(buffer, axis=2).mode
        cv2.imshow("pixel_frequencies", pixel_frequencies.astype(np.uint8))
        return pixel_frequencies

    # Initialize PBAS model
    def pbas_initialize(I, N):
        height, width = I.shape
        samples = np.zeros((height, width, N), dtype=np.uint8)
        for n in range(N):
            x = np.random.randint(max(0, height-1), size=(height, width))
            y = np.random.randint(max(0, width-1), size=(height, width))
            samples[:, :, n] = I[x, y]
        return samples

    # PBAS update function
    def pbas_update(I, samples, R):
        height, width = I.shape
        mask = np.zeros((height, width), dtype=np.uint8)  # Initialize mask
        # Get indices of background pixels
        bg_pixels = np.where(mask == 0)

        # Random subsampling
        rand_samples = np.random.choice(phi, size=len(bg_pixels[0])) == 0
        rand_indices = np.random.choice(N, size=len(bg_pixels[0]))
        samples[bg_pixels[0][rand_samples], bg_pixels[1][rand_samples], rand_indices[rand_samples]] = I[bg_pixels[0][rand_samples], bg_pixels[1][rand_samples]]

        # Random subsampling from neighborhood
        rand_samples = np.random.choice(phi, size=len(bg_pixels[0])) == 0
        x = np.clip(bg_pixels[0] + np.random.randint(-1, 2, size=len(bg_pixels[0])), 0, height-1)
        y = np.clip(bg_pixels[1] + np.random.randint(-1, 2, size=len(bg_pixels[1])), 0, width-1)
        rand_indices = np.random.choice(N, size=len(bg_pixels[0]))
        samples[x[rand_samples], y[rand_samples], rand_indices[rand_samples]] = I[bg_pixels[0][rand_samples], bg_pixels[1][rand_samples]]

        # Update R
        R[bg_pixels] = np.clip(R[bg_pixels] - alpha*(R[bg_pixels] - R_lower), R_lower, R_upper)
        fg_pixels = np.where(mask != 0)
        R[fg_pixels] = np.clip(R[fg_pixels] + alpha*(R[fg_pixels] - R_lower), R_lower, R_upper)

        return mask

    # PBAS subtraction function
    def pbas_subtraction(I, samples, R):
        height, width = I.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # Calculate the absolute difference between I and samples along the third axis
        dist = np.abs(I[:,:,None] - samples)

        # Check if the difference is less than R
        matches = dist < R[:,:,None]

        # Count the number of matches along the third axis
        count = np.sum(matches, axis=2)

        # If pixel is part of the background
        mask[count >= min_matches] = 0
        mask[count < min_matches] = 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 

        return mask

    

    # Function to initialize the buffer
    def initialize_first_buffer(input_dir, buffer, buffer_size, step, roi_start):
        for i in range(roi_start, roi_start + buffer_size, step): 
            # Load the image
            img_path = os.path.join(input_dir, f"in{i:06d}.jpg")
            if not os.path.exists(img_path):
                continue
            I = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if I is None:
                continue
            # Add current frame to the buffer (cyclically)
            buffer.append(I)

    # Function to initialize R matrix
    def initialize_R(height, width):
        return np.ones((height, width), dtype=np.uint8) * R_lower

    # Function to update R matrix
    def update_R(R, mask):
        R = R.astype(np.float64)  # Convert R to float64 for consistent data type
        R[mask == 0] -= alpha * (R[mask == 0] - R_lower)
        R[mask != 0] += alpha * (R_upper - R[mask != 0])
        np.clip(R, R_lower, R_upper, out=R)

    # Function to perform the frequency analysis to determine the background model
    buffer = []
    initialize_first_buffer(input_dir, buffer, buffer_size, step, roi_start)

    # Load the first frame and convert to grayscale
    I = cv2.imread(os.path.join(input_dir, f"in{roi_start:06d}.jpg"), cv2.IMREAD_GRAYSCALE)

    # Perform mode frequency analysis to determine the background model
    bg_model = mode_frequency_analysis(buffer)

    # Initialize PBAS model
    samples = pbas_initialize(bg_model, N)

    # Initialize R matrix
    R = initialize_R(*I.shape)

    # Load ground truth images
    gt_images = load_groundtruth_images(gt_dir, roi_start, roi_end)

    # Process each frame
    for i in range(roi_start + buffer_size, roi_end + 1, step):
        # Load the current frame and convert to grayscale
        I = cv2.imread(os.path.join(input_dir, f"in{i:06d}.jpg"), cv2.IMREAD_GRAYSCALE)

        # Perform PBAS update
        mask = pbas_update(I, samples, R)

        # Perform PBAS subtraction
        mask = pbas_subtraction(I, samples, R)
        update_R(R, mask)
        cv2.imshow('Original Frame', I)
        cv2.imshow('Foreground Mask', mask)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    #vibe_background_subtraction('pedestrian/input', 'pedestrian/groundtruth', roi_start=300, roi_end=1100, N=20, R=20, _min=2, update_factor=16, resize_factor=0.5)
    pbas_background_subtraction('office/input', 'office/groundtruth', roi_start=970, roi_end=2050)
    vibe_background_subtraction('office/input', 'office/groundtruth', roi_start=950, roi_end=2050, N=20, R=20, _min=2, update_factor=16, resize_factor=0.5) # because of randomization it is much slower
    

