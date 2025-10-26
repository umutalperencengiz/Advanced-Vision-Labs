import cv2
import numpy as np
import scipy
import math
from scipy.ndimage import convolve1d,rotate
from sklearn import svm
from sklearn.metrics import confusion_matrix
import scipy.misc 
def load_images(path):
    img = cv2.imread(path)
    return img

def calculate_gradient(img):
    if len(img.shape) == 2:  # Grayscale image
        dx = convolve1d(np.int32(img), np.array([-1, 0, 1]), axis=1)
        dy = convolve1d(np.int32(img), np.array([-1, 0, 1]), axis=0)
        
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * 180 / np.pi
        
    elif len(img.shape) == 3:  # RGB image
        dx_r = convolve1d(np.int32(img[:, :, 0]), np.array([-1, 0, 1]), axis=1)
        dy_r = convolve1d(np.int32(img[:, :, 0]), np.array([-1, 0, 1]), axis=0)
        dx_g = convolve1d(np.int32(img[:, :, 1]), np.array([-1, 0, 1]), axis=1)
        dy_g = convolve1d(np.int32(img[:, :, 1]), np.array([-1, 0, 1]), axis=0)
        dx_b = convolve1d(np.int32(img[:, :, 2]), np.array([-1, 0, 1]), axis=1)
        dy_b = convolve1d(np.int32(img[:, :, 2]), np.array([-1, 0, 1]), axis=0)

        magnitude_r = np.sqrt(dx_r**2 + dy_r**2)
        magnitude_g = np.sqrt(dx_g**2 + dy_g**2)
        magnitude_b = np.sqrt(dx_b**2 + dy_b**2)
        
        max_R = np.logical_and(magnitude_g < magnitude_r, magnitude_b < magnitude_r)
        max_G = np.logical_and(magnitude_r < magnitude_g, magnitude_b < magnitude_g)
        max_B = np.logical_and(magnitude_r < magnitude_b, magnitude_g < magnitude_b)
        
        magnitude = np.zeros_like(magnitude_r)
        orientation = np.zeros_like(magnitude_r)
        
        magnitude[max_R] = magnitude_r[max_R]
        magnitude[max_G] = magnitude_g[max_G]
        magnitude[max_B] = magnitude_b[max_B]
        
        orientation_r = np.arctan2(dy_r, dx_r) * 180 / np.pi
        orientation_g = np.arctan2(dy_g, dx_g) * 180 / np.pi
        orientation_b = np.arctan2(dy_b, dx_b) * 180 / np.pi
        
        orientation[max_R] = orientation_r[max_R]
        orientation[max_G] = orientation_g[max_G]
        orientation[max_B] = orientation_b[max_B]

    return magnitude, orientation

def calculate_histograms(magnitude, orientation, cell_size=8, num_bins=9):
    """Calculate histograms of gradients for cells."""
    YY, XX = magnitude.shape
    YY_cells = np.int32(YY // cell_size)
    XX_cells = np.int32(XX // cell_size)
    
    histograms = np.zeros((YY_cells, XX_cells, num_bins))
    bin_width = 180 / num_bins
    
    for y in range(YY_cells):
        for x in range(XX_cells):
            cell_magnitude = magnitude[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]
            cell_orientation = orientation[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]
            
            for i in range(cell_size):
                for j in range(cell_size):
                    mag = cell_magnitude[i, j]
                    angle = cell_orientation[i, j]

                    if angle < 0:
                        angle += 180  # Adjust negative angles
                   
                    # Determine histogram intervals
                    bin_index = int(angle // bin_width)
                    bin_center = (bin_index + 0.5) * bin_width

                    # Calculate distance from angle to center of interval
                    if bin_index == num_bins - 1:  # Handle the wrapping
                        distance = min(abs(angle - bin_center), abs(180 - angle))
                    else:
                        distance = abs(angle - bin_center)

                    # Ensure bin_index is within the valid range
                    bin_index %= num_bins

                    # Add magnitude to histogram
                    histograms[y, x, bin_index] += mag * (1 - distance / bin_width)
    
    e = math.pow(0.00001, 2)
    normalized_histograms = []
    
    for jj in range(0, YY_cells - 1):
        for ii in range(0, XX_cells - 1):
            H0 = histograms[jj, ii, :]
            H1 = histograms[jj, ii + 1, :]
            H2 = histograms[jj + 1, ii, :]
            H3 = histograms[jj + 1, ii + 1, :]
            H = np.concatenate((H0, H1, H2, H3))
            n = np.linalg.norm(H)
            Hn = H / np.sqrt(math.pow(n, 2) + e)
            normalized_histograms.append(Hn)
      
    return np.concatenate(normalized_histograms)

if __name__ == "__main__":
    path = "pos/per00060.ppm"  # Fix the path
    images = load_images(path)
    magnitude, orientation = calculate_gradient(images)
    histogram = calculate_histograms(magnitude, orientation)
    print(len(histogram))
    print("First 10 elements of normalized histogram:", histogram[:10])
    HOG_data = np.zeros([2 * 100, 3781], np.float32)
    
    # Loop over the images
    for i in range(0, 100):
        # Load positive and negative images
        IP = cv2.imread("pos/per%05d.ppm" % (i + 201))
        IN = cv2.imread("neg/neg%05d.png" % (i + 201))
        
        # Calculate HOG descriptors
        magnitude_pos, orientation_pos = calculate_gradient(IP)
        hog_positive = calculate_histograms(magnitude_pos, orientation_pos)
        magnitude_neg, orientation_neg = calculate_gradient(IN)
        hog_negative = calculate_histograms(magnitude_neg, orientation_neg)
        
        # Assign labels and HOG descriptors to HOG_data
        HOG_data[i, 0] = 1
        HOG_data[i, 1:] = hog_positive
        HOG_data[i + 100, 0] = 0
        HOG_data[i + 100, 1:] = hog_negative
    
    labels = HOG_data[:, 0]
    data = HOG_data[:, 1:]
    
    # Create an SVM classifier
    clf = svm.SVC(kernel='rbf', C=1.0)

    # Train the classifier
    clf.fit(data, labels)
    # Loop over the images
    for i in range(0, 100):
        # Load positive and negative images
        IP = cv2.imread("pos/per%05d.ppm" % (i + 401))
        IN = cv2.imread("neg/neg%05d.png" % (i + 401))
        
        # Calculate HOG descriptors
        magnitude_pos, orientation_pos = calculate_gradient(IP)
        hog_positive = calculate_histograms(magnitude_pos, orientation_pos)
        magnitude_neg, orientation_neg = calculate_gradient(IN)
        hog_negative = calculate_histograms(magnitude_neg, orientation_neg)
        
        # Assign labels and HOG descriptors to HOG_data
        HOG_data[i, 0] = 1
        HOG_data[i, 1:] = hog_positive
        HOG_data[i + 100, 0] = 0
        HOG_data[i + 100, 1:] = hog_negative
    labels = HOG_data[:, 0]
    data = HOG_data[:, 1:]
    lp = clf.predict(data)

    # Analyze the results of learning
    conf_matrix = confusion_matrix(labels, lp)
    TP = conf_matrix[1, 1]  # True positives
    TN = conf_matrix[0, 0]  # True negatives
    FP = conf_matrix[0, 1]  # False positives
    FN = conf_matrix[1, 0]  # False negatives

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy:", accuracy)

    # Load test image
    test_image = cv2.imread("testImage4.png")

    # Calculate HOG descriptors for the test image
    magnitude_test, orientation_test = calculate_gradient(test_image)
    hog_test = calculate_histograms(magnitude_test, orientation_test)

    # Define the sliding window size
    window_size = (64, 128)
    step_size = 16  # Empirical value

    # Loop over the image with sliding window
    detections = []
    for y in range(0, test_image.shape[0] - window_size[1], step_size):
        for x in range(0, test_image.shape[1] - window_size[0], step_size):
            window = test_image[y:y + window_size[1], x:x + window_size[0]]
            magnitude_window, orientation_window = calculate_gradient(window)
            hog_window = calculate_histograms(magnitude_window, orientation_window)
            hog_window = hog_window.reshape(1, -1)  # Reshape to match expected input shape
            prediction = clf.predict(hog_window)
            if prediction == 1:
                detections.append((x, y, x + window_size[0], y + window_size[1]))

    # Visualize detected pedestrian silhouettes
    for (startX, startY, endX, endY) in detections:
        cv2.rectangle(test_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Detected Pedestrian Silhouettes", test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    