    cv2.circle(result_image, (coord[0], coord[1]), 4, (0, 0, 255), -1)
        
        contour = pattern_contours[0]  # Use the first contour
        dy = coord[1] - test_image.shape[0] // 2
        dx = coord[0] - test_image.shape[1] // 2
        cont_new = contour - np.array([dx, dy], dtype=int)
        cv2.drawContours(result_image, [cont_new], -1, (0, 255, 0), 1)