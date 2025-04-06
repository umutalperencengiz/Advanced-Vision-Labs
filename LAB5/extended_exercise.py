import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_pixel(img, center, x, y):
    """Function to get pixel value."""
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except IndexError:
        pass

    return new_value


def lbp_calculated_pixel(img, x, y):
    """Function to calculate LBP pixel value."""
    center = img[x][y]
    val_ar = []

    directions = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
                  (x, y + 1), (x + 1, y + 1), (x + 1, y),
                  (x + 1, y - 1), (x, y - 1)]

    for dx, dy in directions:
        val_ar.append(get_pixel(img, center, dx, dy))

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val 


def BM_depth_map(imgL, imgR):
    stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=19)
    dispmap_bm = stereo_bm.compute(imgL, imgR)
    return dispmap_bm


def SGBM_depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM. Need rectified images, returns depth map (left to right disparity). """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*5,  # Must be divisible by 16
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute the disparity map
    disparity = left_matcher.compute(imgL, imgR).astype(np.float32) / 16.0

    # Normalize the disparity map for visualization
    disparity = (disparity - disparity.min()) / (disparity.max() - disparity.min())
    return disparity


def calculate_lbp(img_bgr):
    height, width, _ = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width), np.uint8)

    for i in range(height):
        for j in range(width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    return img_lbp


def calculate_rms(img1, img2):
    """Calculate the Root Mean Square Error between two images."""
    return np.round(np.sqrt(np.mean((img1 - img2) ** 2)), 4)


def calculate_difference(img1, img2):
    """Calculate the difference between two images."""
    return np.abs(img1 - img2)



if __name__ == "__main__":
    img_l = cv2.imread("aloes/aloeL.jpg")
    img_r = cv2.imread("aloes/aloeR.jpg")
    disp_l = cv2.imread("aloes/dispL.png")
    
    img_lbp = calculate_lbp(img_l)
    
    # Convert images to grayscale
    gray_L = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_R = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    sgbm_disparity = SGBM_depth_map(gray_L, gray_R)
    bm_disparity = BM_depth_map(gray_L, gray_R)
    
    sgbm_disparity = cv2.normalize(sgbm_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    bm_disparity = cv2.normalize(bm_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert the normalized disparity map to a heatmap
    sgbm_heatmap = cv2.applyColorMap(sgbm_disparity, cv2.COLORMAP_HOT)
    bm_heatmap = cv2.applyColorMap(bm_disparity, cv2.COLORMAP_HOT)
    # Calculate the RMS error
    rms_error_BM = calculate_rms(gray_L, bm_disparity)
    rms_error_LBP = calculate_rms(gray_L, img_lbp)
    rms_error_SGBM = calculate_rms(gray_R, sgbm_disparity)

    # Calculate the difference
    difference_BM = calculate_difference(gray_L, bm_disparity)
    difference_LBP = calculate_difference(gray_L, img_lbp)
    difference_SGBM = calculate_difference(gray_R, sgbm_disparity)
    
    # Create a 3x3 subplot grid
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
    
    # Display the original images
    axs[0, 0].imshow(gray_L, cmap='gray')
    axs[0, 0].set_title('Original Image Left')
    axs[0, 1].imshow(disp_l, cmap='gray')
    axs[0, 1].set_title('Reference Disparity Map')
    axs[0, 2].imshow(gray_R, cmap='gray')
    axs[0, 2].set_title('Original Image Right')

    # Display the disparity maps
    axs[1, 0].imshow(bm_disparity, cmap='gray')
    axs[1, 0].set_title(f'BM Disparity Map\nRMS Error: {rms_error_BM}')
    axs[1, 1].imshow(img_lbp, cmap='gray')
    axs[1, 1].set_title(f'LBP Image\nRMS Error: {rms_error_LBP}')
    axs[1, 2].imshow(sgbm_disparity, cmap='gray')
    axs[1, 2].set_title(f'SGBM Disparity Map\nRMS Error: {rms_error_SGBM}')
    
    # Display the difference images
    axs[2, 0].imshow(difference_BM, cmap='gray')
    axs[2, 0].set_title(f'Difference for BM: {np.round(np.mean(difference_BM))}')
    axs[2, 1].imshow(difference_LBP, cmap='gray')
    axs[2, 1].set_title(f'Difference for LBP: {np.round(np.mean(difference_LBP))}')
    axs[2, 2].imshow(difference_SGBM, cmap='gray')
    axs[2, 2].set_title(f'Difference for SGBM: {np.round(np.mean(difference_SGBM))}')

    # Remove axis for all subplots
    for ax in axs.flat:
        ax.axis('off')

    plt.show()
