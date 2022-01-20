import numpy as np
import cv2

def RotationImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    bitwise_not = cv2.bitwise_not(thresh)
    lines = cv2.HoughLinesP(thresh, 1, np.pi, 10, minLineLength=10, maxLineGap=7)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(thresh, (x1, y1), (x2, y2), (255, 255, 255), 2)
    thresh = cv2.bitwise_and(bitwise_not, thresh)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def AreaDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (3, 3))
    (_, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh, None, iterations=1)
    thresh = cv2.subtract(dilate, thresh)
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    temp_c = sorted(cnts, key=cv2.contourArea, reverse=True)
    max = np.max([c.shape[0] for c in temp_c])
    for c in temp_c:
        if c.shape[0] == max:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [box], -1, (255, 255, 255), -1)
            out = cv2.bitwise_not(np.zeros_like(image))
            out[mask == 255] = image[mask == 255]
    return out

def BarcodeDetection(image):
    img = cv2.GaussianBlur(image, (1, 5), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    gray = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
    (_, edges) = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    lines = cv2.HoughLinesP(edges, 1, np.pi, 10, minLineLength=10, maxLineGap=7)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(edges, None, iterations=2)
    cnts, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    temp_c = sorted(cnts, key=cv2.contourArea, reverse=True)
    max = np.max([c.shape[0] for c in temp_c])
    for c in temp_c:
        if c.shape[0]!=max:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(edges, [box], -1, (0, 255, 0), -1)
    return edges

def DrawContours(image,processing_img):
    cnts, hierarchy = cv2.findContours(processing_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    temp_c = sorted(cnts, key=cv2.contourArea, reverse=True)
    max = np.max([c.shape[0] for c in temp_c])

    for c in temp_c:
        if c.shape[0] == max:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            x, y, w, h = cv2.boundingRect(c)
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [box], -1, (255, 255, 255), -1)
            out = image.copy()[y:y+h, x:x+w]
            cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    return image,out

def Main(path_image):
    image = cv2.imread(path_image)
    image = AreaDetection(image)
    image = RotationImage(image)
    houghtransform = BarcodeDetection(image)
    image_detection,barcode = DrawContours(image.copy(),houghtransform)
    return image_detection,barcode

# path = "imgs/images (10).jpg"
# image_detection,barcode = Main(path)
# image = cv2.imread(path)
# cv2.imshow("BarcodeDetection",image_detection)
# barcode = cv2.resize(barcode,None,None,2,2,interpolation=cv2.INTER_CUBIC)
# cv2.imshow("Barcode",barcode)
# cv2.waitKey(0)

