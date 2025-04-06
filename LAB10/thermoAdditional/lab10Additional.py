import cv2
import numpy as np
import matplotlib . pyplot as plt
import os
def process_frame(frame, kernel=np.ones((5, 5), np.uint8)):
    # Thresholding, median filtering, and dilation
    ret, thresholded_img = cv2.threshold(frame, 35, 255, cv2.THRESH_BINARY)
    median_filtered = cv2.medianBlur(thresholded_img, 5)
    dilated = cv2.dilate(median_filtered, kernel, iterations=1)
    """
    # Apply connected components labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8, ltype=cv2.CV_32S)

    # Filter objects based on size and aspect ratio
    min_area_threshold = 80  # Adjust threshold as needed
    aspect_ratio_threshold = 2  # Adjust threshold as needed

    filtered_rectangles = []
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        if w * h > min_area_threshold and w / h < aspect_ratio_threshold:
            filtered_rectangles.append((x, y, w, h))

    # Merge overlapping rectangles
    merged_rectangles = merge_overlapping_rectangles(filtered_rectangles)

    # Draw merged rectangles onto the dilated image
    for rect in merged_rectangles:
        x, y, w, h = rect
        cv2.rectangle(dilated, (x, y), (x + w, y + h), (255, 255, 255), 3)
"""
    return dilated



def load_images(input_dir):
    images = []
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
    return images

def create_probabilistic_pattern(images):
    PDM = np.zeros((192, 64), np.float32)
    for image in images:
        PDM += image
    PDM /= len(images)
    return PDM

iPedestrian = 0
def save_silhouette(ROI, iPedestrian):
    # Resize to 192x64
    resized_ROI = cv2.resize(ROI, (64, 192))
    
    # Check if the bright part is more prominent
    if np.sum(resized_ROI) > np.sum(255 - resized_ROI)*2:
        filename = 'samples/sample_%06d.png' % iPedestrian
        cv2.imwrite(filename, resized_ROI)
        print(f"Saved {filename}")
        return True
    else:
        return False
def find_top_k_maxima(result, k=5, threshold=0.5):
    maxima = []
    result_copy = result.copy()
    for _ in range(k):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_copy)
        if max_val < threshold:
            break
        x, y = max_loc
        w, h = 90, 220  # Assuming the window size
        maxima.append((x, y, w, h))
        
        # Erase the found maximum and its neighbors
        y_start = max(0, y - h // 2)
        y_end = min(result_copy.shape[0], y + h // 2)
        x_start = max(0, x - w // 2)
        x_end = min(result_copy.shape[1], x + w // 2)
        result_copy[y_start:y_end, x_start:x_end] = 0

    return maxima

def detect_objects(frame, PDM1, PDM0):
    h, w = frame.shape
    result = np.zeros((h, w), np.float32)
    window_size = (192, 64)
    
    for y in range(0, h - window_size[0], 4):  # Step size can be adjusted
        for x in range(0, w - window_size[1], 4):
            window = frame[y:y + window_size[0], x:x + window_size[1]]
            B = window / 255.0  # Normalize binary window to 0 and 1
            
            score = np.sum(B * PDM1 + (1 - B) * PDM0)
            result[y:y + window_size[0], x:x + window_size[1]] = score

    result = result / np.max(result)  # Normalize the result
    result_uint8 = np.uint8(result * 255)
    
    maxima = find_top_k_maxima(result, k=30, threshold=0.05)
    return result_uint8, maxima
if __name__ == "__main__":
    """
    cap = cv2 . VideoCapture ("C:\\Users\\umuta\\Documents\\vid1_IR.avi")
    
    while ( cap . isOpened () ):
        ret, frame = cap.read()
        if not ret:  break
        G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = process_frame(G)
        cv2.imshow("IR", processed_frame)

        # Manually select ROI - Here we'll mock a fixed ROI for the example
        y1, y2, x1, x2 = 100, 500, 50, 114 
        ROI = processed_frame[y1:y2, x1:x2]

        # Save the extracted ROI
        save_silhouette(ROI, iPedestrian)
        iPedestrian += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Break the loop when the 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
    cap.release ()
    """
    imagePath = "samples"
    images = load_images(imagePath)
    pattern = create_probabilistic_pattern(images)
    
    plt.imshow(pattern, cmap='gray')
    plt.title("Probabilistic Silhouette Pattern")
    plt.show()
    # Part C: Object detection on a specific frame

    frame_path = "frame_003090.png"
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    processed_frame = process_frame(frame)
    
    # Convert pattern to PDM1 and PDM0
    PDM1 = pattern.astype(np.float32)
    PDM0 = 1 - PDM1
    
    # Detect objects in the processed frame
    result, maxima = detect_objects(processed_frame, PDM1, PDM0)
    
    # Overlay rectangles on the original frame
    frame_with_rectangles = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in maxima:
        cv2.rectangle(frame_with_rectangles, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame with rectangles
    plt.imshow(cv2.cvtColor(frame_with_rectangles, cv2.COLOR_BGR2RGB))
    plt.title("Detected Objects")
    plt.show()
    """
    We can use for non-fixed window size
    Image Pyramid: Use multiple scaled versions of the image (pyramid) and apply your detection method on each scale.
    Multi-scale Sliding Windows: Adjust the window size dynamically to detect objects of different sizes within the original image.
    
    """