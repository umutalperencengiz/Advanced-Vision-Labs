import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib . patches import Rectangle

def rgb2gray (I):
    return 0.299* I [: ,: ,0] + 0.587* I [: ,: ,1] + 0.114* I [: ,: ,2]

def hist(img):
    h=np.zeros ((256 ,1), np.float32) # creates and zeros single -column arrays
    height , width =img.shape [:2] # shape - we take the first 2 values
    for y in range(height):
        for x in range(width):
            intensity = img[y, x]
            h[intensity] += 1
    return h

if __name__ == "__main__":
    #part1.1
    I = cv2.imread("mandrill.jpg")
    cv2.imshow("Mandril I", I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite ("LAB1\mandrill.png ",I )
    print (I.shape ) # dimensions /rows , columns , depth /
    print (I.size ) # number of bytes
    print (I.dtype ) # data type
    #part1.2
    mandril2= plt.imread ("mandrill.jpg")
    plt.figure (1) # create figure
    plt.imshow (mandril2) # add I
    plt.title ("Mandril") # add title
    plt.axis ("off") # disable display of the coordinate system
    plt.show () # display
    plt.imsave ("mandril2.png",mandril2)
    x = [ 100 , 150 , 200 , 250]
    y = [ 50 , 100 , 150 , 200]

    plt.plot(x ,y ,"r.", markersize =10)
    fig,ax = plt.subplots(1) # instead of plt . figure (1)
    rect = Rectangle((50 ,50) ,50 ,100 ,fill = False , ec ="r"); # ec - edge colour
    ax.add_patch(rect) # display
    plt.show()

    #part 1.4
    IG = cv2 . cvtColor (I , cv2 . COLOR_BGR2GRAY )
    IHSV = cv2 . cvtColor (I , cv2 . COLOR_BGR2HSV )
    IH = IHSV [: ,: ,0]
    IS = IHSV [: ,: ,1]
    IV = IHSV [: ,: ,2]
    fig, axes = plt.subplots(1, 3)
  
    axes[0].imshow(IH, cmap='hsv')
    axes[0].set_title('Hue')
    axes[1].imshow(IS, cmap='hsv')
    axes[1].set_title('Saturation')
    axes[2].imshow(IV, cmap='hsv')
    axes[2].set_title('Value')
    plt.show()
    #1.4.2
    I_HSV = matplotlib . colors . rgb_to_hsv (I)
    #1.5 scaling
    height , width =I.shape[:2] # retrieving elements 1 and 2, i.e. the corresponding
    height and width
    scale = 1.75 # scale factor
    Ix2 = cv2 . resize (I ,( int( scale * height ) ,int(scale * width)))
    cv2.imshow(" Big Mandrill ", Ix2 )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #part 1.6 arithmetich
    L = cv2.imread("lena.png")
    I = cv2.resize(I, (480, 480))
    linear_combination = cv2.addWeighted(np.uint8(I), 0.3, np.uint8(L), 0.7, 0)
    
    difference_modulus = cv2.absdiff(np.uint8(I), np.uint8(L))

    #gray Is
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(rgb2gray(I),cmap="gray")
    axes[0].set_title('Mandril Gray')
    axes[1].imshow(rgb2gray(L),cmap="gray")
    axes[1].set_title("Lena Gray")
    plt.show()

    #
    fig, axes = plt.subplots(1,3)
    axes[0].imshow(difference_modulus)
    axes[0].set_title("Substraction L - I")
    axes[1].imshow(cv2.convertScaleAbs(L*0.5))
    axes[1].set_title("Multiplacation L*0.5")
    axes[2].imshow(np.uint8(linear_combination))
    axes[2].set_title("Linear Combination of Lena and Mandrill")
    plt.show()
    # 1.7 Histogram calculation
    fig,axes = plt.subplots(1,2)
    axes[0].hist(cv2.calcHist([IG],[0],None ,[256] ,[0 ,256]))
    axes[0].set_title("CV2 Histogram")
    axes[1].hist(hist(IG))
    axes[1].set_title("Hist Function")
    plt.show()
    #1.8 Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize =(8,8))
    I_CLAHE = clahe.apply(IG)

   
    fig,axes = plt.subplots(1,2)
    axes[0].hist(cv2.equalizeHist(IG))
    axes[0].set_title("CV2 Histogram Equalization")
    axes[1].hist(I_CLAHE)
    axes[1].set_title("Clahe equalization")
    plt.show()
    
        #1.9 Filtration

    gaussian_blur = cv2.GaussianBlur(I, (5, 5), 0)
    sobel_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    laplacian = cv2.Laplacian(I, cv2.CV_64F)
    median_blur = cv2.medianBlur(I, 5)


    plt.figure(figsize=(12, 12))
    plt.subplot(2, 3, 1)
    plt.imshow(I)
    plt.title('Original')

    plt.subplot(2, 3, 2)
    plt.imshow(gaussian_blur)
    plt.title('Gaussian Blur')

    plt.subplot(2, 3, 3)
    plt.imshow(sobel_combined.astype(np.uint8),cmap='gray')
    plt.title('Sobel Filter')

    plt.subplot(2, 3, 4)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Filter')

    plt.subplot(2, 3, 5)
    plt.imshow(median_blur)
    plt.title('Median Blur')

    plt.tight_layout()
    plt.show()


    bilateral_filtered = cv2.bilateralFilter(I, d=9, sigmaColor=75, sigmaSpace=75)
    gabor_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    gabor_filtered = cv2.filter2D(I, cv2.CV_32F, gabor_kernel)


    kernel = np.ones((5, 5), np.uint8)
    eroded_I = cv2.erode(I, kernel, iterations=1)
    dilated_I = cv2.dilate(I, kernel, iterations=1)
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(I)
    plt.title('Original I')

    plt.subplot(2, 3, 2)
    plt.imshow(bilateral_filtered)
    plt.title('Bilateral Filtered I')

    plt.subplot(2, 3, 3)
    plt.imshow(gabor_filtered)
    plt.title('Gabor Filtered I')

    plt.subplot(2, 3, 4)
    plt.imshow(eroded_I)
    plt.title('Eroded I')

    plt.subplot(2, 3, 5)
    plt.imshow(dilated_I)
    plt.title('Dilated I')

    plt.tight_layout()
    plt.show()







