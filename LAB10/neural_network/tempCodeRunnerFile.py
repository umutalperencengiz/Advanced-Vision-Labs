if FUSION == "LATE":
        out_img = img_rgb
        Rect1 = detect(net_therm, img_thermal)
        Rect2 = detect(net_rgb, img_rgb)
        ############ TODO2 ##################
        # "Rect1" i "Rect2" have the format [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
        # 1. Create a list "boxes_iou". Iterating in a double loop through "Rect1" and "Rect2"
        # check the IoU value of each rectangle in these lists (use the IoU() function defined
        # above which takes as arguments the two surrounding rectangles). If the IoU value for a
        # given pair is greater than 0, append to "boxes_iou" a list consisting of a tuple
        # (containing the indices of the currently processed surrounding rectangles) and
        # the calculated IoU value for them.
        # Example: In a given iteration of the double loop, we have reached the 3rd rectangle
        # from "Rect1" and the 4th rectangle from "Rect2". Their common IoU value is 0.55.
        # So we add the list [(3, 4), 0.55] to the array "boxes_iou".
        boxes_iou = []
        for r1 in Rect1:
            for r2 in Rect2:
                IoU_value =IoU(r1,r2)
                if  IoU_value > 0:
                    boxes_iou.append([(r1,r2),IoU_value])

        # 2. Then sort the "boxes_iou" descending by IoU value. Use the sorted() function with the parameters
        # key=lambda a: a[1] oraz reverse=True.
        sorted_boxes_iou = sorted(boxes_iou, key=lambda a: a[1], reverse=True)
        # 3. Create empty lists "Rect1_paired", "Rect2_paired" and "paired_boxes".
        Rect1_paired, Rect2_paired, paired_boxes = [], [], []
        # 4. Create a loop through the elements of "boxes_iou". In each iteration, extract a tuple with index(elem[0])
        # and IoU value(elem[1]) from the currently processed element. If the first index from the tuple is not present
        # in the list "Rect1_paired" and the second element from the tuple is not present in the list "Rect2_paired",
        #  we append the tuple with indices(elem[0]) to "paired_boxes", and append the corresponding indices from the
        # tuple to the lists "Rect1_paired" and "Rect2_paired" (the first to the first list and the second to the second).
        # In this way, we get the list "paired_boxes", which contains pairs of indexes of rectangles from the lists
        # "Rect1" and "Rect2", which need to be paired (average their elements), which will be described in section 5.
        # 5. Finally, we create an empty list of "boxes". Iterating through the tuples in "paired_boxes", we extract from
        # "Rect1" the rectangle with the index stored as the first element of the tuple, and from "Rect2" we extract the
        # rectangle with the index stored as the second element of the tuple. The rectangles are in the form of a 4 element
        # list ([x1, y1, w1, h1]). Having 2 rectangles, i.e. 2 4-element lists (let's call them "r1" and "r2"),
        # we create one new 4-element list (let's call it "avg_r"), whose elements are the average of elements from
        # both lists with rectangles (we remember, that after calculating the average, the result should be converted
        #  to int, avg_r[0] = int((r1[0]/r2[0])/2) and so for all 4 elements. Finally we append "avg_r" to the "boxes" list.
        # This way the format of the "boxes" list will be the same as the format of the lists "Rect1" and "Rect2".
        for elem in sorted_boxes_iou:
            idx1, iou_value = elem
            if idx1 not in Rect1_paired and idx2 not in Rect2_paired:
                paired_boxes.append(elem)
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
            boxes = None

        ######################################
    out_boxes = filter_boxes(boxes)
    for box in out_boxes:
        x, y, w, h = box
        cv2.rectangle(out_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow('Image', out_img)
        cv2.waitKey(10)