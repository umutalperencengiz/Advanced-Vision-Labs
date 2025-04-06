import cv2
import numpy as np

def compute_metrics(object_mask, reference_mask):
    # Compute true positives (TP), false positives (FP), false negatives (FN)
    TP_M = np.logical_and((object_mask == 255), (reference_mask == 255))
    FP_M = np.logical_and((object_mask == 255), (reference_mask == 0))
    FN_M = np.logical_and((object_mask == 0), (reference_mask == 255))

    TP = np.sum(TP_M)
    FP = np.sum(FP_M)
    FN = np.sum(FN_M)
    return TP, FP, FN

if __name__ == "__main__":
    
    TP_total_MOG = 0
    FP_total_MOG = 0
    FN_total_MOG = 0
    TP_total_MOG_special = 0
    FP_total_MOG_special = 0
    FN_total_MOG_special = 0
    TP_total_KNN = 0
    FP_total_KNN = 0
    FN_total_KNN = 0
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    bg_subtractorSpecial = cv2.createBackgroundSubtractorMOG2(varThreshold=15,detectShadows=False)
    bg_subtractor_KNN = cv2.createBackgroundSubtractorKNN(dist2Threshold=70)
    kernel = np.ones((5, 5), np.uint8)
    with open("highway/temporalROI.txt", "r") as f:
        line = f.readline()
        roi_start, roi_end = map(int, line.split())
    for i in range(roi_start,roi_end):
        FI = cv2.imread("highway/input/in%06d.jpg" %i,cv2.IMREAD_GRAYSCALE)
        GI = cv2.imread("highway/groundtruth/gt%06d.png" %i,cv2.IMREAD_GRAYSCALE)

         #3.6  MOG
        fg = bg_subtractor.apply(FI)
        _, thresh_MOG= cv2.threshold(fg, 15, 255, cv2.THRESH_BINARY)
        thresh_subtractor = cv2.morphologyEx(thresh_MOG, cv2.MORPH_OPEN, kernel)
        #Special Values
        fgSpecial= bg_subtractorSpecial.apply(FI)
        _, thresh_MOG_special= cv2.threshold(fgSpecial, 15, 255, cv2.THRESH_BINARY)
        thresh_subtractor_special = cv2.morphologyEx(thresh_MOG_special, cv2.MORPH_OPEN, kernel)
        #3.7 KNN
        fg_KNN = bg_subtractor_KNN.apply(FI)
        _, thresh_KNN = cv2.threshold(fg_KNN, 30, 255, cv2.THRESH_BINARY)
        thresh_subtractor_KNN = cv2.morphologyEx(thresh_KNN, cv2.MORPH_OPEN, kernel)

        TP_MOG, FP_MOG, FN_MOG = compute_metrics(thresh_subtractor, GI)
        TP_total_MOG += TP_MOG
        FP_total_MOG += FP_MOG
        FN_total_MOG += FN_MOG
        TP_MOG_special, FP_MOG_special, FN_MOG_special = compute_metrics(thresh_subtractor_special, GI)
        TP_total_MOG_special += TP_MOG_special
        FP_total_MOG_special += FP_MOG_special
        FN_total_MOG_special += FN_MOG_special
        TP_KNN, FP_KNN, FN_KNN = compute_metrics(thresh_subtractor_KNN, GI)
        TP_total_KNN += TP_KNN
        FP_total_KNN += FP_KNN
        FN_total_KNN += FN_KNN
        cv2.imshow("Background Subtracted MOG", thresh_subtractor)
        cv2.imshow("Background Subtracted MOG Special values", thresh_subtractor_special)
        cv2.imshow("Background Subtracted KNN", thresh_subtractor_KNN)
        cv2.waitKey(10)

    P_total_MOG = TP_total_MOG / (TP_total_MOG + FP_total_MOG) if TP_total_MOG + FP_total_MOG > 0 else 0
    R_total_MOG = TP_total_MOG / (TP_total_MOG + FN_total_MOG) if TP_total_MOG + FN_total_MOG > 0 else 0
    F1_total_MOG = 2 * P_total_MOG * R_total_MOG / (P_total_MOG + R_total_MOG) if P_total_MOG + R_total_MOG > 0 else 0
    print("Overall Precision MOG:", P_total_MOG)
    print("Overall Recall MOG:", R_total_MOG)
    print("Overall F1 Score MOG:", F1_total_MOG)
    P_total_MOG_special = TP_total_MOG_special / (TP_total_MOG_special + FP_total_MOG_special) if TP_total_MOG_special + FP_total_MOG_special > 0 else 0
    R_total_MOG_special = TP_total_MOG_special / (TP_total_MOG_special + FN_total_MOG_special) if TP_total_MOG_special + FN_total_MOG_special > 0 else 0
    F1_total_MOG_special = 2 * P_total_MOG_special * R_total_MOG_special / (P_total_MOG_special + R_total_MOG_special) if P_total_MOG_special + R_total_MOG_special > 0 else 0
    print("Overall Precision MOG Special:", P_total_MOG_special)
    print("Overall Recall MOG Special:", R_total_MOG_special)
    print("Overall F1 Score MOG Special:", F1_total_MOG_special)
    P_total_KNN = TP_total_KNN / (TP_total_KNN + FP_total_KNN) if TP_total_KNN + FP_total_KNN > 0 else 0
    R_total_KNN = TP_total_KNN / (TP_total_KNN + FN_total_KNN) if TP_total_KNN + FN_total_KNN > 0 else 0
    F1_total_KNN = 2 * P_total_KNN * R_total_KNN / (P_total_KNN + R_total_KNN) if P_total_KNN + R_total_KNN > 0 else 0
    print("Overall Precision KNN:", P_total_KNN)
    print("Overall Recall KNN:", R_total_KNN)
    print("Overall F1 Score KNN:", F1_total_KNN)
