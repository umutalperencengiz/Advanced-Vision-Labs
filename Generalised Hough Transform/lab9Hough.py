import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_r_table(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    negated_image = cv2.bitwise_not(binary_image)
    contours= cv2.findContours(negated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
    gradient_magnitude = gradient_magnitude / np.amax(gradient_magnitude)
    
    moments = cv2.moments(negated_image, True)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    
    Rtable = [[] for _ in range(360)]

    for contour in contours:
        for point in contour:
            x, y = point[0]
            gradient_angle = int(gradient_orientation[y, x]) % 360
            r_vector_length = np.sqrt((x - cx)**2 + (y - cy)**2)
            r_vector_angle = np.arctan2(y - cy, x - cx) * 180 / np.pi
            Rtable[gradient_angle].append((r_vector_length, r_vector_angle))
        cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
    return Rtable, contours,img

def generalized_hough_transform(img, Rtable):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_orientation = np.arctan2(sobely, sobelx) * 180 / np.pi
    gradient_magnitude = gradient_magnitude / np.amax(gradient_magnitude)
    
    hough_space = np.zeros_like(gray_image, dtype=np.int32)
    
    for y in range(gradient_magnitude.shape[0]):
        for x in range(gradient_magnitude.shape[1]):
            if gradient_magnitude[y, x] > 0.5:
                gradient_angle = int(gradient_orientation[y, x]) % 360
                for r, theta in Rtable[gradient_angle]:
                    x1 = int(x - r * np.cos(np.deg2rad(theta)))
                    y1 = int(y - r * np.sin(np.deg2rad(theta)))
                    if 0 <= x1 < hough_space.shape[1] and 0 <= y1 < hough_space.shape[0]:
                        hough_space[y1, x1] += 1
    return hough_space

if __name__ == "__main__":
    # Load the pattern image
    pattern_image = cv2.imread('trybik.jpg')
    Rtable, pattern_contours,img = create_r_table(pattern_image)
    """
    cv2.imshow("ss",img)
    cv2.waitKey(0)
    """
    # Load the test image
    test_image = cv2.imread('trybiki2.jpg')
    hough_space = generalized_hough_transform(test_image, Rtable).astype(np.float32)

    # Find the maximum points in the Hough space
    max_coords = []
    window_size = 220

    for i in range(5):
        max_y, max_x = np.where(hough_space == np.max(hough_space))
        max_coords.append((max_x[0], max_y[0]))
        
        # Zero out the surrounding window to find the next maximum
        y_start = max(0, max_y[0] - window_size // 2)
        y_end = min(hough_space.shape[0], max_y[0] + window_size // 2)
        x_start = max(0, max_x[0] - window_size // 2)
        x_end = min(hough_space.shape[1], max_x[0] + window_size // 2)
        hough_space[y_start:y_end, x_start:x_end] = 0
        hough_space = cv2.GaussianBlur(hough_space,(5,5),0)

    cv2.imshow("my", (hough_space * 255).astype(np.float32))
    cv2.waitKey(0)
 

    result_image = test_image.copy()
    contours= pattern_contours

    for coord in max_coords:
        cv2.circle(result_image, (coord[0], coord[1]), 4, (0, 0, 255), -1)
        # Calculate translation
        dy = coord[1] - pattern_image.shape[0] // 2
        dx = coord[0] - pattern_image.shape[1] // 2
        
        for contour in contours:
            # Translate contour points
            translated_contour = contour + np.array([dx, dy])
            # Reshape contour for drawing
            reshaped_contour = translated_contour.reshape((-1, 1, 2))
            cv2.drawContours(result_image, [translated_contour], 0, (255, 0, 0), 3)
        
   

    
    cv2.imshow("Detected Contours", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
