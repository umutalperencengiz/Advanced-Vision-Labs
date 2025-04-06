import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_and_compute_sift(image):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_feature_points(desc1, desc2, k=2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=k)
    best_matches = [[m] for m, n in matches if m.distance < 0.7 * n.distance]
    return best_matches

if __name__ == "__main__":
    # Load images
    fontanna1 = cv2.imread("fontanna1.jpg", cv2.IMREAD_GRAYSCALE)
    fontanna2 = cv2.imread("fontanna2.jpg", cv2.IMREAD_GRAYSCALE)

    # Detect and compute SIFT features
    keypoints1, descriptors1 = detect_and_compute_sift(fontanna1)
    keypoints2, descriptors2 = detect_and_compute_sift(fontanna2)

    # Match SIFT descriptors
    matches = match_feature_points(descriptors1, descriptors2)

    # Draw matches
    img_matches = cv2.drawMatchesKnn(fontanna1, keypoints1, fontanna2, keypoints2, matches, None, flags=2)

    # Display the matches
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches)
    plt.title('SIFT Feature Point Matches')
    plt.axis('off')
    plt.show()
    # this method is better than harris detection