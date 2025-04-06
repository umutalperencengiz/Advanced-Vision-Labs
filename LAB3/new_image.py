import numpy as np
import cv2


beg = 1
end = 1099
step = 1
directory = '/home/vision/Dydaktyka/AVS/Lab 4 - OF'

for i in range(beg, end + 1, step):

    # load img
    I = cv2.imread(directory + "/input0/in%06d.jpg" % i)
    dim = I.shape

    # 352 × 288
    I_big = np.zeros((288, 352, 3), dtype=np.uint8)

    I_big[:240, :352, :] = I[:, :352]

    # cv2.imshow("test", I_big)
    # cv2.waitKey(0)

    cv2.imwrite(directory + "/input_resize/in%06d.jpg" % i, I_big)
    print(directory + "/input_resize/in%06d.jpg" % i)

# cv2.destroyAllWindows()
