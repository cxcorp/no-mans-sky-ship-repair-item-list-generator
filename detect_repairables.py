# -*- coding: utf-8 -*-
import sys
import cv2 as cv
import numpy as np
import math
import imutils
from trackbars import ChangeTrackingTrackbars

IMG_A = "./a.png"

EXIT_KEY = ord('q')

events = [i for i in dir(cv) if 'EVENT' in i]
print(events)


def nothing(x):
    pass


def doThing():
    img = cv.imread(IMG_A)
    template = cv.imread('./broken.png')
    mask = cv.imread('./broken2.png', cv.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit("Couldn't read image")

    b, g, r = cv.split(img)

    #img[:, :, 2] = 0

    img = cv.resize(img, (1920, 1080), interpolation=cv.INTER_LANCZOS4)

    #img = cv.bilateralFilter(img,9, 75, 75)
    #img = cv.GaussianBlur(img, (3, 3), 0)

    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #img = (255-img)
    cv.namedWindow('image')

    tbs = ChangeTrackingTrackbars('image')
    # tbs.add("min").min(0).max(255).initial(0).build()
    # tbs.add("max").min(1).max(255).initial(255).build()
    threshold_max = 700
    tbs.add("threshold").min(0).max(threshold_max).initial(690).build()

    #img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 2)
    #img = cv.Laplacian(img, cv.CV_64F, ksize=7)

    cv.imshow('image', img)

    w, h = mask.shape[::-1]

    while True:
        if tbs.changed:
            # print('changed')
            tbs.changed = False

            result = img.copy()
            
            thresholdI = tbs.getPos("threshold")
            threshold = thresholdI / float(threshold_max)

            res = cv.matchTemplate(result, template, cv.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)

            result = cv.cvtColor(cv.cvtColor(result, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

            for pt in zip(*loc[::-1]):
                cv.rectangle(result, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

            cv.putText(result, f'threshold: {threshold:.3f}', (15, 35), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

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
