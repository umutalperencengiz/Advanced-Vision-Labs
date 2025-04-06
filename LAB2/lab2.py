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
    IP = cv2.imread ("pedestrian/input/in%06d.jpg" % 1 )
    IP= cv2.cvtColor(IP,cv2.COLOR_BGR2GRAY)
    TP_total = 0
    FP_total = 0
    FN_total = 0
    with open("pedestrian/temporalROI.txt", "r") as f:
        line = f.readline()
        roi_start, roi_end = map(int, line.split())
    for i in range (300,1099) :

        Input = cv2.imread ("pedestrian/input/in%06d.jpg" % i )
        I = cv2.cvtColor(Input,cv2.COLOR_BGR2GRAY)
        # Load reference mask
        reference_mask = cv2.imread("pedestrian/groundtruth/gt%06d.png" % i,cv2.IMREAD_GRAYSCALE)
        substraction = cv2.absdiff(I,IP)
        (thresh, ImageOutput) = cv2.threshold(substraction ,15,255 ,cv2.THRESH_BINARY )
        
        kernel = np.ones((8,8),np.uint8)
        ImageOutput = cv2.medianBlur(ImageOutput,17)
        opening = cv2.morphologyEx(ImageOutput, cv2.MORPH_OPEN, kernel)
        
        retval , labels , stats , centroids = cv2.connectedComponentsWithStats (ImageOutput)
        if ( stats . shape [0] > 1) : # are there any objects
            tab = stats [1: ,4] # 4 columns without first element
            pi = np . argmax ( tab )# finding the index of the largest item
            pi = pi + 1 # increment because we want the index in stats , not in tab
            # drawing a bbox
            cv2 . rectangle ( ImageOutput ,( stats [pi ,0] , stats [pi ,1]) ,( stats [ pi ,0]+ stats [ pi ,2] , stats [ pi
            ,1]+ stats [ pi ,3]) ,(255 ,0 ,0) ,2)
            # print information about the field and the number of the largest element
            cv2.putText(ImageOutput, "%d" % pi, (np.int32(centroids[pi, 0]), np.int32(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            cv2 . putText ( ImageOutput ,"%d" %pi ,( np.int32( centroids [ pi ,0]) ,np.int32( centroids [ pi ,1]) ) , cv2.FONT_HERSHEY_SIMPLEX ,1 ,(255 ,0 ,0) )
        cv2.imshow ("Image output", ImageOutput )
        cv2 . waitKey(10)
        TP, FP, FN = compute_metrics(ImageOutput, reference_mask)
        TP_total += TP
        FP_total += FP
        FN_total += FN

    P_total = TP_total / (TP_total + FP_total) if TP_total + FP_total > 0 else 0
    R_total = TP_total / (TP_total + FN_total) if TP_total + FN_total > 0 else 0
    F1_total = 2 * P_total * R_total / (P_total + R_total) if P_total + R_total > 0 else 0

    print("Overall Precision:", P_total)
    print("Overall Recall:", R_total)
    print("Overall F1 Score:", F1_total)
    
    IP = I
