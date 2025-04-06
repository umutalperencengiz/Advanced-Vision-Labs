import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

def calibrate_and_rectify_fisheye(image_dir, direction, width, height, square_size, img_width, img_height, number_of_images, noDetection=[]):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

    objp = np.zeros((height * width, 1, 3), np.float32)
    objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []

    for i in range(1, number_of_images):
        if i in noDetection:
            print(f"Skipping image pair {i} due to previous detection failure.")
            continue

        img = cv2.imread(image_dir + direction + "_%02d.png" % i)
        if img is None:
            print(f"Image {image_dir + direction + '_%02d.png' % i} not found")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (width, height),
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print(f"Chessboard couldn't be detected in {direction} image pair {i}")
            noDetection.append(i)
            continue

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print(f"No valid detections found for {direction} images.")
        return None, None, None, None, None, [], [], noDetection

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]

    ret, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        (img_width, img_height),
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    if not ret:
        print(f"Calibration failed for {direction} images.")
        return None, None, None, None, None, [], [], noDetection

    print(f"Camera calibration successful for {direction} images. Calibration error: {ret}")

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (img_width, img_height), cv2.CV_16SC2)

    return img, map1, map2, K, D, imgpoints, objpoints, noDetection

def BM_depth_map(imgL, imgR):
    stereo_bm = cv2.StereoBM_create(numDisparities=64, blockSize=19)
    dispmap_bm = stereo_bm.compute(imgL, imgR)
    return dispmap_bm

def SGBM_depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM. Need rectified images, returns depth map (left to right disparity). """
    window_size = 3

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


def main():
    image_dir = "pairs/"
    width = 9
    height = 6
    square_size = 0.025
    img_width = 640
    img_height = 480
    number_of_images = 50

    img_l, map1_left, map2_left, K_left, D_left, imgpointsLeft, objpoints, noDetectionsLeft = calibrate_and_rectify_fisheye(
        image_dir, "left", width, height, square_size, img_width, img_height, number_of_images)

    img_r, map1_right, map2_right, K_right, D_right, imgpointsRight, objpointsRight, noDetectionsRight = calibrate_and_rectify_fisheye(
        image_dir, "right", width, height, square_size, img_width, img_height, number_of_images, noDetectionsLeft)

    if img_l is None or img_r is None:
        print("Calibration failed for one or both cameras.")
        exit(1)

    if len(imgpointsLeft) < 1 or len(imgpointsRight) < 1:
        print("Not enough valid points for stereo calibration.")
        exit(1)
    
    imgpointsLeft = np.asarray(imgpointsLeft, dtype=np.float64)
    imgpointsRight = np.asarray(imgpointsRight, dtype=np.float64)
    objpoints = np.asarray(objpoints, dtype=np.float64)

    valid_pairs = min(len(imgpointsLeft), len(imgpointsRight))
    imgpointsLeft = imgpointsLeft[:valid_pairs]
    imgpointsRight = imgpointsRight[:valid_pairs]
    objpoints = objpoints[:valid_pairs]

    (RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpointsLeft,
        imgpointsRight,
        K_left,
        D_left,
        K_right,
        D_right,
        (img_width, img_height),
        None,
        None,
        cv2.CALIB_FIX_INTRINSIC,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
    )

    print(f"Stereo calibration RMS error: {RMS}")

    R2 = np.zeros([3, 3])
    P1 = np.zeros([3, 4])
    P2 = np.zeros([3, 4])
    Q = np.zeros([4, 4])

    (leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap) = cv2.fisheye.stereoRectify(
        K_left, D_left,
        K_right, D_right,
        (img_width, img_height),
        rotationMatrix, translationVector,
        0, R2, P1, P2, Q,
        cv2.CALIB_ZERO_DISPARITY, (0, 0), 0, 0)

    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, leftRectification,
        leftProjection, (img_width, img_height), cv2.CV_16SC2)

    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, rightRectification,
        rightProjection, (img_width, img_height), cv2.CV_16SC2)

    img = cv2.imread("example/example0.jpg")
    
    height, width = img.shape[:2]
    img_l = img[:, :width // 2]
    img_r = img[:, width // 2:]
    
    img_l = cv2.resize(img_l, (img_width, img_height))
    img_r = cv2.resize(img_r, (img_width, img_height))
    
    cv2.imshow("Left Image", img_l)
    cv2.imshow("Right Image", img_r)
    cv2.waitKey(0)

    dst_L = cv2.remap(img_l, map1_left, map2_left, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dst_R = cv2.remap(img_r, map1_right, map2_right, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow("Rectified Left Image", dst_L)
    cv2.imshow("Rectified Right Image", dst_R)
    cv2.waitKey(0)

    dst_L = cv2.cvtColor(dst_L, cv2.COLOR_BGR2GRAY)
    dst_R = cv2.cvtColor(dst_R, cv2.COLOR_BGR2GRAY)
    
    dispmap_bm = BM_depth_map(dst_L, dst_R)
    dispmap_sgbm = SGBM_depth_map(dst_L, dst_R)

    plt.subplot(121)
    plt.imshow(dispmap_bm, cmap='gray')
    plt.title('BM Disparity')

    plt.subplot(122)
    plt.imshow(dispmap_sgbm, cmap='gray')
    plt.title('SGBM Disparity')

    plt.show()

if __name__ == "__main__":
    main()
