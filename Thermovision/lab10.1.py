import cv2
import numpy as np
import matplotlib . pyplot as plt

def merge_overlapping_rectangles(rectangles):
    merged_rectangles = []

    # Sort rectangles by their Y coordinate
    sorted_rectangles = sorted(rectangles, key=lambda rect: rect[1])

    # Iterate through sorted rectangles
    i = 0
    while i < len(sorted_rectangles):
        current_rect = sorted_rectangles[i]

        # Check if there's a next rectangle to compare
        if i < len(sorted_rectangles) - 1:
            next_rect = sorted_rectangles[i + 1]
            # Check if the next rectangle overlaps vertically with the current one
            if current_rect[1] <= next_rect[1] + next_rect[3]:  # Check if bottom of current rect is above top of next rect
                # Merge the rectangles
                merged_rect = (min(current_rect[0], next_rect[0]),  # Left
                               min(current_rect[1], next_rect[1]),  # Top
                               max(current_rect[0] + current_rect[2], next_rect[0] + next_rect[2]) - min(current_rect[0], next_rect[0]),  # Width
                               max(current_rect[1] + current_rect[3], next_rect[1] + next_rect[3]) - min(current_rect[1], next_rect[1]))  # Height
                # Replace the current rectangle with the merged rectangle
                sorted_rectangles[i] = merged_rect
                # Skip the next rectangle as it's now part of the merged one
                i += 1
        # Add the current (or merged) rectangle to the list of merged rectangles
        merged_rectangles.append(sorted_rectangles[i])
        i += 1

    return merged_rectangles

def process_frame(frame, kernel=np.ones((5, 5), np.uint8)):
    # Thresholding, median filtering, and dilation
    ret, thresholded_img = cv2.threshold(frame, 35, 255, cv2.THRESH_BINARY)
    median_filtered = cv2.medianBlur(thresholded_img, 5)
    dilated = cv2.dilate(median_filtered, kernel, iterations=1)

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

    return dilated

if __name__ == "__main__":
    cap = cv2 . VideoCapture ("C:\\Users\\umuta\\Documents\\vid1_IR.avi")
    
    while ( cap . isOpened () ):
        ret , frame = cap.read ()
        G = cv2.cvtColor( frame , cv2 . COLOR_BGR2GRAY )
        processed_frame = process_frame(G)
        cv2.imshow ("IR ", processed_frame)
        if cv2 . waitKey (1) & 0xFF == ord ("q"): # break the loop when the ’q’ key
            break
    cap.release ()
