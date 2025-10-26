import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_fast_detector(image, threshold=60, nonmaxSuppression=True):
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmaxSuppression)
    keypoints = fast.detect(image, None)
    return keypoints

def compute_harris_response(image, ksize=3, k=0.04):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=ksize)
    Ixx = dx * dx
    Iyy = dy * dy
    Ixy = dx * dy
    Sxx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigmaX=1.5)
    Syy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigmaX=1.5)
    Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigmaX=1.5)
    det = (Sxx * Syy) - (Sxy * Sxy)
    trace = Sxx + Syy
    R = det - k * (trace ** 2)
    
    return R

def get_top_n_keypoints(keypoints, R, N):
    keypoints_with_response = [(kp, R[int(kp.pt[1]), int(kp.pt[0])]) for kp in keypoints]
    keypoints_with_response.sort(key=lambda x: x[1], reverse=True)
    top_n_keypoints = [kp[0] for kp in keypoints_with_response[:N]]
    return top_n_keypoints

def compute_centroid_orientation(image, keypoints, patch_size=31):
    half_size = patch_size // 2
    centroids_orientations = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = image[max(0, y-half_size):min(image.shape[0], y+half_size+1), 
                      max(0, x-half_size):min(image.shape[1], x+half_size+1)]
        moments = cv2.moments(patch)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            orientation = 0.5 * np.arctan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])
        else:
            cx, cy, orientation = 0, 0, 0
        centroids_orientations.append((kp.pt[0], kp.pt[1], orientation))
    return centroids_orientations

def load_pairs(filename):
    pairs = []
    with open(filename, 'r') as file:
        for line in file:
            values = list(map(float, line.split()))
            if len(values) == 4:
                x1, y1, x2, y2 = values
                pairs.append((x1, y1, x2, y2))
            else:
                print(f"Skipping line: {line.strip()} - does not contain exactly 4 values")
    return pairs

def compute_brief_descriptor(image, keypoints, pairs, angles):
    descriptors = []
    for kp, angle in zip(keypoints, angles):
        patch_size = 31
        half_size = patch_size // 2
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = image[max(0, y-half_size):min(image.shape[0], y+half_size+1), 
                      max(0, x-half_size):min(image.shape[1], x+half_size+1)]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            continue
        blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
        descriptor = []
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        for (x1, y1, x2, y2) in pairs:
            x1_prime = cos_angle * x1 - sin_angle * y1
            y1_prime = sin_angle * x1 + cos_angle * y1
            x2_prime = cos_angle * x2 - sin_angle * y2
            y2_prime = sin_angle * x2 + cos_angle * y2
            if 0 <= x1_prime + half_size < patch_size and 0 <= y1_prime + half_size < patch_size and \
               0 <= x2_prime + half_size < patch_size and 0 <= y2_prime + half_size < patch_size:
                intensity1 = blurred_patch[int(y1_prime + half_size), int(x1_prime + half_size)]
                intensity2 = blurred_patch[int(y2_prime + half_size), int(x2_prime + half_size)]
                if intensity1 < intensity2:
                    descriptor.append(1)
                else:
                    descriptor.append(0)
        descriptors.append(np.array(descriptor, dtype=np.uint8))
    return descriptors

def match_descriptors(descriptors1, descriptors2):
    matches = []
    for i, d1 in enumerate(descriptors1):
        min_dist = np.inf
        best_match = None
        for j, d2 in enumerate(descriptors2):
            if len(d1) == len(d2):
                # Compute Hamming distance
                dist = np.sum(d1 != d2)
                if dist < min_dist:
                    min_dist = dist
                    best_match = cv2.DMatch(i, j, float(dist))
        if best_match is not None:
            matches.append(best_match)
    return matches

def filter_keypoints_within_bounds(image, keypoints, patch_size=31):
    half_size = patch_size // 2
    filtered_keypoints = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if x - half_size >= 0 and x + half_size < image.shape[1] and y - half_size >= 0 and y + half_size < image.shape[0]:
            filtered_keypoints.append(kp)
    return filtered_keypoints

if __name__ == "__main__":
    fontanna1 = cv2.imread("fontanna1.jpg", cv2.IMREAD_GRAYSCALE)
    fontanna2 = cv2.imread("fontanna2.jpg", cv2.IMREAD_GRAYSCALE)
    if fontanna1 is None or fontanna2 is None:
        print("Error: Could not load images.")
    else:
        # Detect keypoints using FAST
        fast_threshold = 70
        fontanna1_keypoints = filter_keypoints_within_bounds(fontanna1, apply_fast_detector(fontanna1, threshold=fast_threshold))
        fontanna2_keypoints = filter_keypoints_within_bounds(fontanna2, apply_fast_detector(fontanna2, threshold=fast_threshold))

        # Compute the Harris response
        harris_response1 = compute_harris_response(fontanna1)
        harris_response2 = compute_harris_response(fontanna2)

        # Sort and select the top N keypoints
        N = 2800
        top_n_keypoints1 = get_top_n_keypoints(fontanna1_keypoints, harris_response1, N)
        top_n_keypoints2 = get_top_n_keypoints(fontanna2_keypoints, harris_response2, N)

        # Compute centroids and orientations
        centroids_orientations1 = compute_centroid_orientation(fontanna1, top_n_keypoints1)
        centroids_orientations2 = compute_centroid_orientation(fontanna2, top_n_keypoints2)

        # Extract keypoints and orientations
        keypoints1 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in centroids_orientations1]
        angles1 = [kp[2] for kp in centroids_orientations1]
        keypoints2 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in centroids_orientations2]
        angles2 = [kp[2] for kp in centroids_orientations2]

        # Load pairs of points for BRIEF
        pairs = load_pairs('orb_descriptor_positions.txt')

        # Compute BRIEF descriptors
        descriptors1 = compute_brief_descriptor(fontanna1, keypoints1, pairs, angles1)
        descriptors2 = compute_brief_descriptor(fontanna2, keypoints2, pairs, angles2)

        # Match descriptors
        matches = match_descriptors(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        top_matches_num = 240
        # Draw matches
        img_matches = cv2.drawMatches(fontanna1, keypoints1, fontanna2, keypoints2, matches[:top_matches_num], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.title('ORB Matches')
        plt.axis('off')
        plt.show()
    """
    #because of images in eiffel there is improvement but still is not very useful for matching
    eiffel1 = cv2.imread("eiffel1.jpg", cv2.IMREAD_GRAYSCALE)
    eiffel2 = cv2.imread("eiffel2.jpg", cv2.IMREAD_GRAYSCALE)

    if eiffel1 is None or eiffel2 is None:
        print("Error: Could not load images.")
    else:
        # Compute Harris responses for Eiffel Tower images
        filter_size = 3
        eiffel1_harris = compute_harris_response(eiffel1, ksize=filter_size)
        eiffel2_harris = compute_harris_response(eiffel2, ksize=filter_size)

        # Detect keypoints using FAST
        fast_threshold = 70
        eiffel1_keypoints = filter_keypoints_within_bounds(eiffel1, apply_fast_detector(eiffel1, threshold=fast_threshold))
        eiffel2_keypoints = filter_keypoints_within_bounds(eiffel2, apply_fast_detector(eiffel2, threshold=fast_threshold))

        # Sort and select the top N keypoints
        N = 5000
        top_n_keypoints1 = get_top_n_keypoints(eiffel1_keypoints, eiffel1_harris, N)
        top_n_keypoints2 = get_top_n_keypoints(eiffel2_keypoints, eiffel2_harris, N)

        # Compute centroids and orientations
        centroids_orientations1 = compute_centroid_orientation(eiffel1, top_n_keypoints1)
        centroids_orientations2 = compute_centroid_orientation(eiffel2, top_n_keypoints2)

        # Extract keypoints and orientations
        keypoints1 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in centroids_orientations1]
        angles1 = [kp[2] for kp in centroids_orientations1]
        keypoints2 = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in centroids_orientations2]
        angles2 = [kp[2] for kp in centroids_orientations2]
        descriptors1 = compute_brief_descriptor(eiffel1, keypoints1, pairs, angles1)
        descriptors2 = compute_brief_descriptor(eiffel2, keypoints2, pairs, angles2)
        # Match descriptors
        matches = match_descriptors(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw matches
        img_matches = cv2.drawMatches(eiffel1, keypoints1, eiffel2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_matches)
        plt.title('ORB Matches')
        plt.axis('off')
        plt.show()
        
    """