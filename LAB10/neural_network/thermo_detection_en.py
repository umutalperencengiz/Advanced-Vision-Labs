from os.path import join
import cv2
import numpy as np

############################### Functions #################################


def detect(net, img):
    size = img.shape
    height = size[0]
    width = size[1]
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
    return boxes


def filter_boxes(boxes):
    all_paired_boxes = list()
    for ii, box1 in enumerate(boxes):
        x1, y1, w1, h1 = box1
        center_x1 = x1 + int(w1 / 2)
        center_y1 = y1 + int(h1 / 2)
        to_connect = [ii]
        for jj, box2 in enumerate(boxes):
            if jj != ii:
                x2, y2, w2, h2 = box2
                center_x2 = x2 + int(w2 / 2)
                center_y2 = y2 + int(h2 / 2)
                if abs(center_x2 - center_x1) < 10 and abs(center_y2 - center_y1) < 10:
                    to_connect.append(jj)
        all_paired_boxes.append(to_connect)
    all_paired_boxes = sorted(all_paired_boxes, key=lambda x: len(x), reverse=True)
    all_paired = list()
    final_boxes = list()
    for conn in all_paired_boxes:
        if all([a not in all_paired for a in conn]):
            for a in conn:
                all_paired.append(a)
            final_boxes.append(conn)
    out_boxes = [[int(sum([boxes[i][a] for i in elem]) / len(elem)) for a in range(4)] for elem in final_boxes]
    return out_boxes


def IoU(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    left = max([x1, x2])
    right = min([x1+w1, x2+w2])
    top = max([y1, y2])
    bottom = min([y1+h1, y2+h2])
    area1 = max([(right - left), 0]) * max([(bottom - top), 0])
    area2 = (w1 * h1) + (w2 * h2) - area1
    IoU = area1/area2
    return IoU


############# METHOD ###############
# Choose your fusion method
FUSION = "LATE"
#FUSION = "EARLY"

############# TODO0 ###############
# Set the path
test_rgb = r"C:\Users\umuta\Documents\test_rgb"  
test_thermal = r"C:\Users\umuta\Documents\test_thermal"  
###################################

net_fus = None
net_therm = None
net_rgb = None
if FUSION == "EARLY":
    net_fus = cv2.dnn.readNet('yolov3_training_last_f.weights', 'yolov3.cfg')
if FUSION == "LATE":
    #net_therm = cv2.dnn.readNet('neural_network/yolov3_training_last_t.weights', 'neural_network/yolov3_t.cfg')
    net_therm = cv2.dnn.readNet('yolov3_training_last_t.weights', 'yolov3_t.cfg')
    net_rgb = cv2.dnn.readNet('yolov3_training_last_c.weights', 'yolovo3_c.cfg')

for i in range(200, 300):  # you can change the range up to 518
    path_rgb = join(test_rgb, f"img{i}.png")
    path_thermal = join(test_thermal, f"img{i}.png")
    img_rgb = cv2.imread(path_rgb)
    img_thermal = cv2.imread(path_thermal)
    img_thermal = cv2.cvtColor(img_thermal, cv2.COLOR_BGR2GRAY)
    out_img = None
    boxes = None
    if FUSION == "EARLY":
        ############ TODO1 ##################
        # Combine RGB with Thermal by following the instructions
        # 1. Create a new frame (numpy array) with the dimensions of an RGB image and call it new_fus
        new_fus = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), dtype=np.uint8)
        # 2. Copy the first two channels of the RGB (img_rgb[:, :, :2]) to the first two channels of the new frame (new_fus[:, :, :2])
        new_fus[:,:,:2] = img_rgb[:, :, :2]
        # 3. The value of the third channel of the new frame is the maximum of the value of the third RGB channel and the thermal image (single channel)
        #    Use for example np.maximum(a, b). Where a and b are the 3rd RGB channel and thermal
        new_fus[:,:,2] = np.maximum(img_rgb[:,:,2], img_thermal)
        
        # 4. Convert "new_fus" to "uint8" (new_fus.astype("uint8"))
        new_fus = new_fus.astype("uint8")
        boxes = detect(net_fus, new_fus)
        out_img = new_fus
        new_fus = None

        ####################################
        
        
    if FUSION == "LATE":
        out_img = img_rgb
        Rect1 = detect(net_therm, img_thermal)
        Rect2 = detect(net_rgb, img_rgb)

        boxes_iou = []
        for i, r1 in enumerate(Rect1):
            for j, r2 in enumerate(Rect2):
                IoU_value = IoU(r1, r2)
                if IoU_value > 0:
                    boxes_iou.append([(i, j), IoU_value])

        sorted_boxes_iou = sorted(boxes_iou, key=lambda a: a[1], reverse=True)

        Rect1_paired, Rect2_paired, paired_boxes = [], [], []

        for elem in sorted_boxes_iou:
            (idx1, idx2), iou_value = elem
            if idx1 not in Rect1_paired and idx2 not in Rect2_paired:
                paired_boxes.append((idx1, idx2))
                Rect1_paired.append(idx1)
                Rect2_paired.append(idx2)

        boxes = []
        for idx1, idx2 in paired_boxes:
            r1 = Rect1[idx1]
            r2 = Rect2[idx2]
            avg_r = [
                int((r1[0] + r2[0]) / 2),
                int((r1[1] + r2[1]) / 2),
                int((r1[2] + r2[2]) / 2),
                int((r1[3] + r2[3]) / 2)
            ]
            boxes.append(avg_r)


        ######################################
    out_boxes = filter_boxes(boxes)
    for box in out_boxes:
        x, y, w, h = box
        cv2.rectangle(out_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow('Image', out_img)
        cv2.waitKey(10)
