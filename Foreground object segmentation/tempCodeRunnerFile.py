for i in range(roi_start-60,roi_end):
        FI = cv2.imread("pedestrian/input/in%06d.jpg" %i,cv2.IMREAD_GRAYSCALE)
        GI = cv2.imread("pedestrian/groundtruth/gt%06d.png" %i,cv2.IMREAD_GRAYSCALE)