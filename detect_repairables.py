# -*- coding: utf-8 -*-
import sys
import cv2 as cv
import numpy as np
import math
import imutils
from trackbars import ChangeTrackingTrackbars

IMG_A = "./a.png"

EXIT_KEY = ord('q')

import cv2 as cv
events = [i for i in dir(cv) if 'EVENT' in i]
print( events )

def nothing(x):
    pass

def doThing():
    img = cv.imread(IMG_A)
    if img is None:
        sys.exit("Couldn't read image")

    b, g, r = cv.split(img)

    img[:, :, 2] = 0

    img = cv.resize(img, (1920, 1080), interpolation=cv.INTER_LANCZOS4)

    #img = cv.bilateralFilter(img,9, 75, 75)
    img = cv.GaussianBlur(img, (5, 5), 0)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #img = (255-img)
    tb_rho = 'rho'
    tb_thresh = 'threshold'

    cv.namedWindow('image')

    tbs = ChangeTrackingTrackbars('image')
    tbs.add(tb_rho).min(1).max(200).initial(1).build()
    tbs.add(tb_thresh).min(1).max(255).initial(200).build()
    tbs.add('min_theta').min(0).max(180).initial(0).build()
    tbs.add('max_theta').min(0).max(180).initial(180).build()
    def validateTrackbars(x):
        min_theta = tbs.getPos('min_theta')
        max_theta = tbs.getPos('max_theta')
        if max_theta < min_theta:
            tbs.setPos('max_theta', min_theta)
    tbs.setOnChange(validateTrackbars)

    mouseChanged = False
    mouseDown = False
    clickStart = [0, 0]
    mousePos = [0, 0]

    mouseRoi = None
    roiChanged = False

    def mouseCallback(event, x, y, flags, param):
        nonlocal mouseDown, clickStart, mousePos, mouseChanged, mouseRoi, roiChanged

        if event == cv.EVENT_LBUTTONDOWN:
            mouseChanged = True
            mouseDown = True
            clickStart[0] = x
            clickStart[1] = y
        elif event == cv.EVENT_LBUTTONUP:
            mouseChanged = True
            mouseDown = False
            mouseRoi = (clickStart.copy(), mousePos.copy())
            roiChanged = True
        elif event == cv.EVENT_MOUSEMOVE:
            if mouseDown:
                mousePos[0] = x
                mousePos[1] = y
                mouseChanged = True
        elif event == cv.EVENT_LBUTTONDBLCLK:
            mouseChanged = True
            mouseDown = False
            mouseRoi = None
            roiChanged = True
            
    cv.setMouseCallback('image', mouseCallback)

    #img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 2)
    #img = cv.Laplacian(img, cv.CV_64F, ksize=7)

    cv.imshow('image', img)

    imgWidth = img.shape[1]

    prevImg = img.copy()
    frame = 0

    while True:
        if tbs.changed or roiChanged:
            #print('changed')
            tbs.changed = False
            roiChanged = False

            thresh = tbs.getPos(tb_thresh)
            rho = tbs.getPos(tb_rho)
            min_theta = tbs.getPos('min_theta')
            max_theta = tbs.getPos('max_theta')

            src = img
            if mouseRoi is not None:
                start, end = mouseRoi
                yFrom = min(start[1], end[1])
                yTo = max(start[1], end[1])
                xFrom = min(start[0], end[0])
                xTo = max(start[0], end[0])
                src = img[yFrom:yTo,xFrom:xTo]

            result = cv.Canny(src,
                              20,
                              80,
                              apertureSize=3)

            one_deg_in_rad = np.pi / 180
            lines = cv.HoughLines(result, rho, one_deg_in_rad, thresh, min_theta=min_theta*one_deg_in_rad, max_theta=max_theta*one_deg_in_rad)

            result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

            if lines is not None:
                for line in lines:
                    rh, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rh
                    y0 = b*rh
                    x1 = int(x0 + imgWidth*2*(-b))
                    y1 = int(y0 + imgWidth*2*(a))
                    x2 = int(x0 - imgWidth*2*(-b))
                    y2 = int(y0 - imgWidth*2*(a))
                    cv.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2, cv.LINE_AA)
            
            if mouseRoi is not None:
                start, end = mouseRoi
                yFrom = min(start[1], end[1])
                yTo = max(start[1], end[1])
                xFrom = min(start[0], end[0])
                xTo = max(start[0], end[0])

                roi = result
                result = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                
                result[yFrom:yTo,xFrom:xTo] = roi

            cv.putText(result, f'rho: {rho}', (20, 50),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv.putText(result, f'thresh: {thresh}', (20, 90),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            frame = frame+1
            cv.putText(result, f'{frame}', (20, 130),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            prevImg = result.copy()

            #result = cv.GaussianBlur(img, (sobel, sobel), 0)
            cv.imshow('image', result)
        elif mouseChanged:
            mouseChanged = False
            print('mc')
            result = prevImg.copy()
            if mouseDown:
                cv.rectangle(result, clickStart, mousePos, (0, 0, 255), 1, cv.LINE_4)
            cv.imshow('image', result)

        k = cv.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            break

    # ret, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)

    # img = (255-img)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
    # #mg = cv.dilate(img, kernel, iterations=1)

    # cv.imshow("nms", img)
    # while cv.getWindowProperty("nms", 0) >= 0:
    #     key = cv.waitKey(0) & 0xFF
    #     if key == EXIT_KEY:
    #         break
    cv.destroyAllWindows()
    sys.exit("exiting")


doThing()
