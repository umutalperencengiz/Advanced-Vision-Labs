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

def mean_approximation(IN, BGN_prev,alpha = 0.01):
    return alpha * IN + (1 - alpha) * BGN_prev

def median_approximation(IN, BGN_prev):
    result = np.where(BGN_prev < IN, BGN_prev + 1, np.where(BGN_prev > IN, BGN_prev - 1, BGN_prev))
    return result


if __name__ == "__main__":
    
    with open("pedestrian/temporalROI.txt", "r") as f:
        line = f.readline()
        roi_start, roi_end = map(int, line.split())
    start= roi_start -60
    F1I = cv2.imread("pedestrian/input/in%06d.jpg" % start ,cv2.IMREAD_GRAYSCALE)
    BGN_mean = F1I.astype(np.float64)
    BGN_median = F1I.copy()
    BGN_meanc = F1I.astype(np.float64)
    BGN_medianc = F1I.copy()
    YY, XX = F1I.shape
    N = 60
    BUF = np.zeros((YY, XX, N), dtype=np.uint8)
    iN = 0
    TP_total_mean = 0
    FP_total_mean = 0
    FN_total_mean = 0
    TP_total_median = 0
    FP_total_median = 0
    FN_total_median = 0
    TP_sg_total_mean = 0
    FP_sg_total_mean = 0
    FN_sg_total_mean = 0
    TP_sg_total_median = 0
    FP_sg_total_median = 0
    FN_sg_total_median = 0
    TP_sgc_total_mean = 0
    FP_sgc_total_mean = 0
    FN_sgc_total_mean = 0
    TP_sgc_total_median = 0
    FP_sgc_total_median = 0
    FN_sgc_total_median = 0
    prev_mask_mean = None
    prev_mask_median = None
   
    for i in range(roi_start-60,roi_end):
        FI = cv2.imread("pedestrian/input/in%06d.jpg" %i,cv2.IMREAD_GRAYSCALE)
        GI = cv2.imread("pedestrian/groundtruth/gt%06d.png" %i,cv2.IMREAD_GRAYSCALE)
        BUF[:, :, iN] = FI
        iN = (iN + 1) % N
        mean_buffer = np.mean(BUF, axis=2).astype(np.uint8)
        median_buffer = np.median(BUF, axis=2).astype(np.uint8)
        # Background subtraction
        diff_mean = cv2.absdiff(FI, mean_buffer)
        diff_median = cv2.absdiff(FI, median_buffer)

        # Binarization (you may need to adjust the threshold value)
        _, thresh_mean = cv2.threshold(diff_mean, 30, 255, cv2.THRESH_BINARY)
        _, thresh_median = cv2.threshold(diff_median,30 , 255, cv2.THRESH_BINARY)

        # Optional: Morphological operations for noise removal
        kernel = np.ones((5, 5), np.uint8)
        thresh_mean = cv2.morphologyEx(thresh_mean, cv2.MORPH_OPEN, kernel)
        thresh_median = cv2.morphologyEx(thresh_median, cv2.MORPH_OPEN, kernel)
        #3.4 Sigma Delta
        diff_mean_sg = cv2.absdiff(FI,BGN_mean.astype(np.uint8))
        diff_median_sg = cv2.absdiff(FI, BGN_median)
        _, thresh_mean_sg = cv2.threshold(diff_mean_sg, 30, 255, cv2.THRESH_BINARY)
        _, thresh_median_sg = cv2.threshold(diff_median_sg, 30, 255, cv2.THRESH_BINARY)
        thresh_mean_sg = cv2.morphologyEx(thresh_mean_sg, cv2.MORPH_OPEN, kernel)
        thresh_median_sg = cv2.morphologyEx(thresh_median_sg, cv2.MORPH_OPEN, kernel)
        BGN_mean = mean_approximation(FI,BGN_mean)
        BGN_median = median_approximation(FI,BGN_median)
        # 3.5 Conservative update approach
        diff_mean_sgc = cv2.absdiff(FI,BGN_meanc.astype(np.uint8))
        diff_median_sgc = cv2.absdiff(FI, BGN_medianc)
        _, thresh_mean_sgc = cv2.threshold(diff_mean_sgc, 30, 255, cv2.THRESH_BINARY)
        _, thresh_median_sgc = cv2.threshold(diff_median_sgc, 30, 255, cv2.THRESH_BINARY)
        thresh_mean_sgc = cv2.morphologyEx(thresh_mean_sgc, cv2.MORPH_OPEN, kernel)
        thresh_median_sgc = cv2.morphologyEx(thresh_median_sgc, cv2.MORPH_OPEN, kernel)
        if prev_mask_mean is not None: BGN_meanc[prev_mask_mean == 255] = mean_approximation(FI[prev_mask_mean == 255], BGN_mean[prev_mask_mean == 255])
        if prev_mask_median is not None: BGN_medianc[prev_mask_median == 255] = mean_approximation(FI[prev_mask_median == 255], BGN_mean[prev_mask_median == 255])
        # Update previous mask
        prev_mask_mean = thresh_mean_sgc
        prev_mask_median = thresh_median_sgc
     
        
        TP_mean, FP_mean, FN_mean = compute_metrics(thresh_mean, GI)
        TP_total_mean += TP_mean
        FP_total_mean += FP_mean
        FN_total_mean += FN_mean
        TP_median, FP_median, FN_median = compute_metrics(thresh_median, GI)
        TP_total_median += TP_median
        FP_total_median += FP_median
        FN_total_median += FN_median
        TP_sg_mean, FP_sg_mean, FN_sg_mean = compute_metrics(thresh_mean_sg, GI)
        TP_sg_total_mean += TP_sg_mean
        FP_sg_total_mean += FP_sg_mean
        FN_sg_total_mean += FN_sg_mean
        TP_sg_median, FP_sg_median, FN_sg_median = compute_metrics(thresh_median_sg, GI)
        TP_sg_total_median += TP_sg_median
        FP_sg_total_median += FP_sg_median
        FN_sg_total_median += FN_sg_median
        TP_sgc_mean, FP_sgc_mean, FN_sgc_mean = compute_metrics(thresh_mean_sgc, GI)
        TP_sgc_total_mean += TP_sgc_mean
        FP_sgc_total_mean += FP_sgc_mean
        FN_sgc_total_mean += FN_sgc_mean
        TP_sgc_median, FP_sgc_median, FN_sgc_median = compute_metrics(thresh_median_sgc, GI)
        TP_sgc_total_median += TP_sgc_median
        FP_sgc_total_median += FP_sgc_median
        FN_sgc_total_median += FN_sgc_median
        cv2.imshow("BUFFER", thresh_mean)
        cv2.waitKey(10)
       
        
        


    P_total_mean = TP_total_mean / (TP_total_mean + FP_total_mean) if TP_total_mean + FP_total_mean > 0 else 0
    R_total_mean = TP_total_mean / (TP_total_mean + FN_total_mean) if TP_total_mean + FN_total_mean > 0 else 0
    F1_total_mean = 2 * P_total_mean * R_total_mean / (P_total_mean + R_total_mean) if P_total_mean + R_total_mean > 0 else 0
    print("Overall Precision Mean:", P_total_mean)
    print("Overall Recall Mean:", R_total_mean)
    print("Overall F1 Score Mean:", F1_total_mean)
    P_total_median = TP_total_median / (TP_total_median + FP_total_median) if TP_total_median + FP_total_median > 0 else 0
    R_total_median = TP_total_median / (TP_total_median + FN_total_median) if TP_total_median + FN_total_median > 0 else 0
    F1_total_median = 2 * P_total_median * R_total_median / (P_total_median + R_total_median) if P_total_median + R_total_median > 0 else 0
    print("\nOverall Precision Median:", P_total_median)
    print("Overall Recall Median:", R_total_median)
    print("Overall F1 Score Median:", F1_total_median)
    P_sg_total_mean = TP_sg_total_mean / (TP_sg_total_mean + FP_sg_total_mean) if TP_sg_total_mean + FP_sg_total_mean > 0 else 0
    R_sg_total_mean = TP_sg_total_mean / (TP_sg_total_mean + FN_sg_total_mean) if TP_sg_total_mean + FN_sg_total_mean > 0 else 0
    F1_sg_total_mean = 2 * P_sg_total_mean * R_sg_total_mean / (P_sg_total_mean + R_sg_total_mean) if P_sg_total_mean + R_sg_total_mean > 0 else 0
    print("\nOverall Precision Sigma-Delta Mean:", P_sg_total_mean)
    print("Overall Recall Sigma-Delta Mean:", R_sg_total_mean)
    print("Overall F1 Sigma-Delta Score Mean:", F1_sg_total_mean)
    P_sg_total_median = TP_sg_total_median / (TP_sg_total_median + FP_sg_total_median) if TP_sg_total_median + FP_sg_total_median > 0 else 0
    R_sg_total_median = TP_sg_total_median / (TP_sg_total_median + FN_sg_total_median) if TP_sg_total_median + FN_sg_total_median > 0 else 0
    F1_sg_total_median = 2 * P_sg_total_median * R_sg_total_median / (P_sg_total_median + R_sg_total_median) if P_sg_total_median + R_sg_total_median > 0 else 0
    print("\nOverall Precision Sigma-Delta Median:", P_sg_total_median)
    print("Overall Recall Sigma-Delta Median:", R_sg_total_median)
    print("Overall F1 Score Sigma-Delta Median:", F1_sg_total_median)
    P_sgc_total_mean = TP_sgc_total_mean / (TP_sgc_total_mean + FP_sgc_total_mean) if TP_sgc_total_mean + FP_sgc_total_mean > 0 else 0
    R_sgc_total_mean = TP_sgc_total_mean / (TP_sgc_total_mean + FN_sgc_total_mean) if TP_sgc_total_mean + FN_sgc_total_mean > 0 else 0
    F1_sgc_total_mean = 2 * P_sgc_total_mean * R_sgc_total_mean / (P_sgc_total_mean + R_sgc_total_mean) if P_sgc_total_mean + R_sgc_total_mean > 0 else 0
    print("\nOverall Precision Conservative way Sigma-Delta Mean:", P_sgc_total_mean)
    print("Overall Recall Conservative way Sigma-Delta Mean:", R_sgc_total_mean)
    print("Overall F1 Conservative way Sigma-Delta Score Mean:", F1_sgc_total_mean)
    P_sgc_total_median = TP_sgc_total_median / (TP_sgc_total_median + FP_sgc_total_median) if TP_sgc_total_median + FP_sgc_total_median > 0 else 0
    R_sgc_total_median = TP_sgc_total_median / (TP_sgc_total_median + FN_sgc_total_median) if TP_sgc_total_median + FN_sgc_total_median > 0 else 0
    F1_sgc_total_median = 2 * P_sgc_total_median * R_sgc_total_median / (P_sgc_total_median + R_sgc_total_median) if P_sgc_total_median + R_sgc_total_median > 0 else 0
    print("\nOverall Precision Conservative way Sigma-Delta Median:", P_sgc_total_median)
    print("Overall Recall Conservative way Sigma-Delta Median:", R_sgc_total_median)
    print("Overall F1 Score Conservative way Sigma-Delta Median:", F1_sgc_total_median)
 