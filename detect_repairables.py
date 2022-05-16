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


def rgb2lab(rgb):
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0:1, 0:1] = rgb

    lab = cv.cvtColor(img, cv.COLOR_RGB2Lab)
    return lab[0][0]

templates = [
    "exploded_panel"
]

def doThing():
    img = cv.imread(IMG_A)
    templateImages = [(cv.imread(f'./images/{template}.png'), template) for template in templates]

    if img is None:
        sys.exit("Couldn't read image")

    #img = cv.resize(img, (255, 1080), interpolation=cv.INTER_LANCZOS4)

    cv.namedWindow('image')

    tbs = ChangeTrackingTrackbars('image')
    maxThreshold = 700
    # tbs.add("maskThreshold").min(1).max(255).initial(200).build()
    # tbs.add("blockSize").min(1).max(100).initial(2).build()
    tbs.add("threshold").min(0).max(maxThreshold).initial(maxThreshold).build()

    cv.imshow('image', img)

    blurred = cv.GaussianBlur(img, (3, 3), 0)
    lab = cv.cvtColor(blurred, cv.COLOR_BGR2Lab)

    redSample = (123, 0, 0)
    redSampleLab = rgb2lab(redSample)

    redTreshold = 4

    maskThreshold = 200 #tbs.getPos("maskThreshold")

    lower = np.clip([x - redTreshold for x in redSampleLab], 0, 255)
    upper = np.clip([x + redTreshold for x in redSampleLab], 0, 255)
    mask = cv.inRange(lab, lower, upper)

    mask = cv.dilate(mask, cv.getStructuringElement(
        cv.MORPH_RECT, (maskThreshold, maskThreshold)))
    
    img = cv.bitwise_and(img, img, mask=mask)

    while True:
        if tbs.changed:
            # print('changed')
            tbs.changed = False

            #result = img.copy()
            threshold = 0.9 #tbs.getPos("threshold") / float(maxThreshold)
            result = img.copy()

            for template, templateName in templateImages:
                w, h, _ = template.shape
                res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)

                loc = np.where(res >= threshold)
                for pt in zip(*loc[::-1]):
                    cv.rectangle(result, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 2)
                    cv.putText(result, templateName, (pt[0], pt[1]), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 5)
                    cv.putText(result, templateName, (pt[0], pt[1]), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

            
            #result = cv.bitwise_and(result, result, mask=mask)

            cv.putText(result, f'threshold: {threshold:.3f}', (15, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            cv.imshow('image', result)

        k = cv.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            break

    cv.destroyAllWindows()
    sys.exit("exiting")


doThing()
