import numpy as np
import cv2

def count_children(hierarchy, parent, inner=False):
    if parent == -1:
        return 0
    elif not inner:
        return count_children(hierarchy, hierarchy[parent][2], True)
    return 1 + count_children(hierarchy, hierarchy[parent][0], True) + count_children(hierarchy, hierarchy[parent][2], True)

def AreaDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(9, 9))
    gray = clahe.apply(gray)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    (_, thresh) = cv2.threshold(gradient, 150, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
    out = None
    cnts, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    temp_c = sorted(cnts, key=cv2.contourArea, reverse=True)

    areaDetect = np.zeros_like(dilate)

    if len(temp_c) != 0:
        temp_c_filter = [temp for temp in temp_c if temp.shape[0] > 10]
        for rect in temp_c_filter:
            rect = cv2.minAreaRect(rect)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(areaDetect,[box],-1,(255,255,255),-1)
    areaDetect = cv2.dilate(areaDetect, None, iterations=15)
    checkareaDetect = cv2.bitwise_and(areaDetect,np.bitwise_not(dilate))
    cnts,_ = cv2.findContours(areaDetect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    _, hierarchy = cv2.findContours(checkareaDetect.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    detect_c = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(detect_c) != 0:
        i=0
        for detect in detect_c:
            if count_children(hierarchy[0],i) >= 35 and detect.shape[0] == detect_c[0].shape[0]:
                x, y, w, h = cv2.boundingRect(detect)
                rect = cv2.minAreaRect(detect)
                box = np.int0(cv2.boxPoints(rect))
                out = image.copy()[y:y + h, x:x + w]
                cv2.drawContours(image,[box],-1,(0,255,0),2)
            i+=1
    return image,out