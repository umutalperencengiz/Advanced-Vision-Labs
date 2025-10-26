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

def load_sequence_info(sequence_folder):
    with open(f"{sequence_folder}/temporalROI.txt", "r") as f:
        line = f.readline()
        roi_start, roi_end = map(int, line.split())
    return roi_start, roi_end

def load_frame_sequence(sequence_folder, roi_start, buffer_size):
    frame_shape = cv2.imread(f"{sequence_folder}/input/in000001.jpg", cv2.IMREAD_GRAYSCALE).shape
    BUF = np.zeros((*frame_shape, buffer_size), dtype=np.uint8)
    for i in range(roi_start - buffer_size, roi_start):
        F1I = cv2.imread(f"{sequence_folder}/input/in%06d.jpg" % i, cv2.IMREAD_GRAYSCALE)
        BUF[:, :, i - (roi_start - buffer_size)] = F1I
    return BUF

def compute_most_frequent_value(BUF):
    frequency = np.zeros((256, BUF.shape[0], BUF.shape[1]), dtype=np.uint8)
    for i in range(BUF.shape[2]):
        for j in range(BUF.shape[0]):
            for k in range(BUF.shape[1]):
                pixel_value = BUF[j, k, i]
                frequency[pixel_value, j, k] += 1

    most_frequent_index = np.argmax(frequency, axis=0)
    most_frequent_value = np.zeros_like(most_frequent_index)
    for j in range(most_frequent_index.shape[0]):
        for k in range(most_frequent_index.shape[1]):
            most_frequent_value[j, k] = BUF[j, k, min(most_frequent_index[j, k], BUF.shape[2] - 1)]
    return most_frequent_value.astype(np.uint8)

def apply_background_subtraction(frame, background_model):
    diff = cv2.absdiff(frame, background_model)
    _, thresholded_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    processed_diff = cv2.morphologyEx(thresholded_diff, cv2.MORPH_OPEN, kernel)
    return processed_diff

if __name__ == "__main__":
    sequences = ["pedestrian", "highway", "office"]
    for seq in sequences:
        roi_start, roi_end = load_sequence_info(seq)
        BUF = load_frame_sequence(seq, roi_start, buffer_size=30)
        background_model = compute_most_frequent_value(BUF)

        for i in range(roi_start, roi_end):
            FI = cv2.imread(f"{seq}/input/in%06d.jpg" % i, cv2.IMREAD_GRAYSCALE)

            diff_mean = apply_background_subtraction(FI, background_model)
            cv2.imshow("Background Subtraction", diff_mean)
            cv2.waitKey(10)