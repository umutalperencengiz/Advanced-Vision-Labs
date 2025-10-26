import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_resize_images(left_path, right_path):
    left_panorama = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_panorama = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    common_height = min(left_panorama.shape[0], right_panorama.shape[0])
    left_resized = cv2.resize(left_panorama, (int(left_panorama.shape[1] * common_height / left_panorama.shape[0]), common_height))
    right_resized = cv2.resize(right_panorama, (int(right_panorama.shape[1] * common_height / right_panorama.shape[0]), common_height))
    return left_resized, right_resized

def find_keypoints(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_keypoints(left_keypoints, left_descriptors, right_keypoints, right_descriptors):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(left_descriptors, right_descriptors)
    ptsA = np.float32([left_keypoints[m.queryIdx].pt for m in matches])
    ptsB = np.float32([right_keypoints[m.trainIdx].pt for m in matches])
    return ptsA, ptsB

def calculate_homography(ptsA, ptsB):
    H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)
    return H

def warp_and_stitch_images(left_image, right_image, H):
    width = left_image.shape[1] + right_image.shape[1]
    height = left_image.shape[0]
    warped_left = cv2.warpPerspective(left_image, H, (width, height))
    stitched_image = warped_left.copy()
    stitched_image[0:right_image.shape[0], 0:right_image.shape[1]] = right_image
    return stitched_image

def remove_excess_background(image):
    mask = image != 0
    cropped_image = image[np.ix_(mask.any(1), mask.any(0))]
    return cropped_image

def main(left_path, right_path):
    left_resized, right_resized = load_and_resize_images(left_path, right_path)
    left_keypoints, left_descriptors = find_keypoints(left_resized)
    right_keypoints, right_descriptors = find_keypoints(right_resized)
    ptsA, ptsB = match_keypoints(left_keypoints, left_descriptors, right_keypoints, right_descriptors)
    H = calculate_homography(ptsA, ptsB)
    stitched_image = warp_and_stitch_images(left_resized, right_resized, H)
    cropped_image = remove_excess_background(stitched_image)
    plt.imshow(cropped_image, cmap='gray')
    plt.title('Stitched and Cropped Panorama')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main("left_panorama.jpg", "right_panorama.jpg")
