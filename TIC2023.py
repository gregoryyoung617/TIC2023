import cv2
import numpy as np
import time
import math



# Change the frame settings and print results to console
#configure for live / not live
is_run_as_loop = True
pic_path = 'red2.jpg'



print (f"loop: {is_run_as_loop}")
looping = True

if is_run_as_loop:
    sleep_time = 0.2
else:
    sleep_time = 7


if is_run_as_loop:
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print ("Changing frame settings:")
    print (cap.set(3, 1280.0))
    print (cap.set(4, 960.0))

# Run in a loop until the user decides to exit
while looping:
    # Capture frame-by-frame
   
    if is_run_as_loop:
        ret, frame = cap.read()
    else:
        looping = False
        ret = True
        frame = cv2.imread(pic_path,cv2.IMREAD_UNCHANGED)
    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    black = cv2.imread("pure-white-background-85a2a7fd.jpg")   

    newgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    alpha = 3 # Contrast control
    beta = 5 # Brightness control

    # call convertScaleAbs function
    adjusted = cv2.convertScaleAbs(newgray, alpha=alpha, beta=beta)

    kernel = np.float32([[0,0,0],[-2,1,-1],[1.2,1.3,0.7]])
    img_out = cv2.filter2D(adjusted,-1,kernel)

    thresh = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, blockSize=15, C=9)

    # Invert the binary image
    thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((4,3), np.uint8)

    eroded_img = cv2.erode(thresh, kernel, iterations=6)

    newdilated_img = cv2.dilate(thresh,kernel, iterations=1)

    final_image = newdilated_img

    circles = cv2.HoughCircles(final_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=10, param2=20, minRadius=10, maxRadius=40)
    cnt = 0
   
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles: 
            is_overlapping = False
            for (x1, y1, r1) in circles:
                
                distance = math.sqrt((x - x1)**2 + (y - y1)**2)

                # Check if the circles overlap
                if is_overlapping or ((x1==x) and (y1==y) and (r1==r)):
                    continue
                elif distance < (r1 + r):
                    is_overlapping = True
                else:
                    continue
            if is_overlapping:
                g = 0
                red = 255
            else:
                g= 255
                red = 0

            cv2.circle(frame, (x, y), r, (0,g,red), 2)
            cnt += 1

    cv2.imshow('TIC', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

    time.sleep(sleep_time)

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()