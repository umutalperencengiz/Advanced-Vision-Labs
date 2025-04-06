import cv2
import numpy as np
import scipy.ndimage.filters as filters
from matplotlib import pyplot as plt
import pm

def harris_method(image, filter_size):
    # Apply Sobel filter to get derivatives
    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=filter_size)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=filter_size)
    
    # Compute products of derivatives
    Ixx = sobel_x * sobel_x
    Iyy = sobel_y * sobel_y
    Ixy = sobel_x * sobel_y

    # Apply Gaussian filter to smooth the products
    Sxx = cv2.GaussianBlur(Ixx, (filter_size, filter_size), sigmaX=1.5)
    Syy = cv2.GaussianBlur(Iyy, (filter_size, filter_size), sigmaX=1.5)
    Sxy = cv2.GaussianBlur(Ixy, (filter_size, filter_size), sigmaX=1.5)

    # Compute the Harris response
    k = 0.05
    det = (Sxx * Syy) - (Sxy * Sxy)
    trace = Sxx + Syy
    R = det - k * (trace ** 2)

    # Normalize the response to range 0-1
    R = cv2.normalize(R, None, 0, 1, cv2.NORM_MINMAX)
    return R

def find_max(image, size, threshold):
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)

def extract_patches(image, keypoints, patch_size):
    height, width = image.shape
    half_size = patch_size // 2

    # Filter keypoints to ensure the patch fits in the image
    valid_keypoints = list(filter(
        lambda pt: pt[0] >= half_size and pt[0] < height - half_size and pt[1] >= half_size and pt[1] < width - half_size, 
        zip(keypoints[0], keypoints[1])
    ))
    patches = []
    for y, x in valid_keypoints:
        patch = image[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
        patches.append((patch.flatten(), (y,x)))
    return patches

def match_descriptors(descriptors1, descriptors2):
    """Match descriptors using Normalized Cross-Correlation (NCC)."""
    # Extract patch arrays from the list of descriptors
    patches1 = np.array([patch[0] for patch in descriptors1])
    patches2 = np.array([patch[0] for patch in descriptors2])
    
    # Normalize patches
    patches1 = (patches1 - patches1.mean(axis=1)[:, None]) / patches1.std(axis=1)[:, None]
    patches2 = (patches2 - patches2.mean(axis=1)[:, None]) / patches2.std(axis=1)[:, None]
    
    # Compute NCC between descriptors
    ncc_scores = np.dot(patches1, patches2.T)
    
    # Find the best match for each descriptor
    matches = []
    for i in range(len(patches1)):
        max_score_index = np.argmax(ncc_scores[i])
        matches.append(cv2.DMatch(i, max_score_index, 1 - ncc_scores[i][max_score_index]))  # Inverse of NCC score for distance
    return matches



def find_n_similar_points(patches1, patches2, n):
    """Find and display the top n similar points between two images."""
    matches = match_descriptors(patches1, patches2)
    matches = sorted(matches, key=lambda x: x.distance )
    top_n_matches = matches[:n]
    return top_n_matches


if __name__ == "__main__":
    # Read the images in grayscale
    fontanna1 = cv2.imread("fontanna1.jpg", cv2.IMREAD_GRAYSCALE)
    fontanna2 = cv2.imread("fontanna2.jpg", cv2.IMREAD_GRAYSCALE)
    budynek1 = cv2.imread("budynek1.jpg",cv2.IMREAD_GRAYSCALE)
    budynek2 = cv2.imread("budynek2.jpg",cv2.IMREAD_GRAYSCALE)
    eiffel1 = cv2.imread("eiffel1.jpg",cv2.IMREAD_GRAYSCALE)
    eiffel2 = cv2.imread("eiffel2.jpg",cv2.IMREAD_GRAYSCALE)

    if fontanna1 is None or fontanna2 is None:
        print("Error: Could not load images.")
    else:
        # Apply Harris corner detection
        filter_size = 9
        harris_response1 = harris_method(fontanna1, filter_size)
        harris_response2 = harris_method(fontanna2, filter_size)
        budynek1_harris = harris_method(budynek1,filter_size)
        budynek2_harris = harris_method(budynek2,filter_size)
        eiffel1_harris = harris_method(eiffel1,filter_size)
        eiffel2_harris = harris_method(eiffel2,filter_size)
        # Find local maxima in the Harris response
        maxima1_y, maxima1_x = find_max(harris_response1, size=9, threshold=0.38)
        maxima2_y, maxima2_x = find_max(harris_response2, size=9, threshold=0.36)
        maxima3_y, maxima3_x = find_max(budynek1_harris, size=9, threshold=0.40)
        maxima4_y, maxima4_x = find_max(budynek2_harris, size=9, threshold=0.32)
        maxima5_y, maxima5_x = find_max(eiffel1_harris, size=9, threshold=0.36)
        maxima6_y, maxima6_x = find_max(eiffel2_harris, size=9, threshold=0.36)
        #Feature Extraction and Vectorisation
        patch_size = 30  # You can change this to any desired size
        patches1= extract_patches(fontanna1, (maxima1_y, maxima1_x), patch_size)
        patches2= extract_patches(fontanna2, (maxima2_y, maxima2_x), patch_size)
        patches3= extract_patches(budynek1, (maxima3_y, maxima3_x), patch_size)
        patches4= extract_patches(budynek2, (maxima4_y, maxima4_x), patch_size)
        patches5= extract_patches(eiffel1, (maxima5_y, maxima5_x), patch_size)
        patches6= extract_patches(eiffel2, (maxima6_y, maxima6_x), patch_size)
        """
        # Display the results
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title('Harris Response - Fontanna1')
        plt.imshow(fontanna1, cmap='gray')
        plt.scatter(maxima1_x, maxima1_y, c='r', s=10)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Harris Response - Fontanna2')
        plt.imshow(fontanna2, cmap='gray')
        plt.scatter(maxima2_x, maxima2_y, c='r', s=10)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        plt.subplot(1, 2, 1)
        plt.title('Harris Response - Budynek1')
        plt.imshow(budynek1, cmap='gray')
        plt.scatter(maxima3_x, maxima3_y, c='r', s=10)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Harris Response - Budynek2')
        plt.imshow(budynek2, cmap='gray')
        plt.scatter(maxima4_x, maxima4_y, c='r', s=10)
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        """
        n = 25
        # Find and display the top 25 similar points for fontanna images
        top_n_matches_fontanna = find_n_similar_points(patches1, patches2, n)
        # Convert cv2.DMatch objects to the required format
        top_n_matches_converted = [(patches1[match.queryIdx][1], patches2[match.trainIdx][1]) for match in top_n_matches_fontanna]
        pm.plot_matches(fontanna1, fontanna2, top_n_matches_converted)

        # Find and display the top 25 similar points for budynek images
        top_n_matches_budynek = find_n_similar_points(patches3, patches4, n)
        top_n_matches_converted2 = [(patches3[match.queryIdx][1], patches4[match.trainIdx][1]) for match in top_n_matches_budynek]
        pm.plot_matches(budynek1, budynek2, top_n_matches_converted2)
        
        # Find and display the top 25 similar points for eiffel images 
        top_n_matches_eiffel = find_n_similar_points(patches5, patches6, n)
        top_n_matches_converted3 = [(patches5[match.queryIdx][1], patches6[match.trainIdx][1]) for match in top_n_matches_eiffel]
        pm.plot_matches(eiffel1, eiffel2, top_n_matches_converted3) # in eiffel image it does not work perfectly because of brightness 
        
    