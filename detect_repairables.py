# -*- coding: utf-8 -*-
from operator import index
import sys
import cv2 as cv
from cv2 import blur
import numpy as np
import math
import imutils
from matplotlib import pyplot as plt
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
    "burnt_out_compressor",
    "containn",
    "corroded_tanks",
    "damaged_gears",
    "damaged_hydraylics",
    "exploded_panel",
    "hull_fracture",
    "melted_fuel_cell",
    "radiation_leak",
    "rusted",
    "scorched_plating",
    "shattered"
]


def doThing():
    img = cv.imread(IMG_A)
    templateImages = [
        (cv.imread(f'./images/{template}.png'), template) for template in templates]

    if img is None:
        sys.exit("Couldn't read image")

    #img = cv.resize(img, (255, 1080), interpolation=cv.INTER_LANCZOS4)

    cv.namedWindow('image')

    tbs = ChangeTrackingTrackbars('image')
    # tbs.add("maskThreshold").min(1).max(255).initial(200).build()
    # tbs.add("blockSize").min(1).max(100).initial(2).build()
    tbs.add("block").min(3).max(49).initial(3).build()
    disti_max = 700
    tbs.add("dist").min(0).max(disti_max).initial(10).build()

    cv.imshow('image', img)

    # blurred = cv.GaussianBlur(img, (3, 3), 0)
    # lab = cv.cvtColor(blurred, cv.COLOR_BGR2Lab)

    # redSample = (123, 0, 0)
    # redSampleLab = rgb2lab(redSample)

    # redTreshold = 4

    # maskThreshold = 200 #tbs.getPos("maskThreshold")

    # lower = np.clip([x - redTreshold for x in redSampleLab], 0, 255)
    # upper = np.clip([x + redTreshold for x in redSampleLab], 0, 255)
    # mask = cv.inRange(lab, lower, upper)

    # mask = cv.dilate(mask, cv.getStructuringElement(
    #     cv.MORPH_RECT, (maskThreshold, maskThreshold)))

    # img = cv.bitwise_and(img, img, mask=mask)

    #img = cv.medianBlur(img, ksize=3)
    

    click = None

    def mouseListener(event, x, y, flags, param):
        nonlocal click, tbs
        if event == cv.EVENT_LBUTTONDOWN:
            click = (x, y)
            tbs.changed = True

    cv.setMouseCallback('image', mouseListener)


    img = cv.GaussianBlur(img, (5, 5), 0)
    b, g, r = cv.split(img)

    imgWidth,imgHeight,_ = img.shape

    while True:
        if tbs.changed:
            # print('changed')
            tbs.changed = False

            # offset = tbs.getPos("offset")
            # lower = np.clip([x - offset for x in graySample], 0, 255)
            # upper = np.clip([x + offset for x in graySample], 0, 255)
            # grayMask = cv.inRange(lab, lower, upper)
            #result = img.copy()
            block = tbs.getPos("block")
            if block % 2 == 0:
                block = block + 1

            #img = [r,g,b][rgb]
            #img = (255-img.copy())

            # 3 or 9

            # a1 = cv.adaptiveThreshold(
            #     (255-g), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block, 2)
            # a2 = cv.adaptiveThreshold(
            #     (255-b), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block, 2)
            # img = cv.bitwise_or(a1, a2)
            img = g.copy()
            img = cv.Canny(img, 20, 80)

            contours, hierarchy = cv.findContours(
                img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

            distRel = tbs.getPos("dist") / float(disti_max)
            dist = imgHeight * distRel


            #ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
            cv.putText(img, f'dist%: {distRel:.3f}', (438, 395),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv.putText(img, f'dist: {dist:.3f}', (438, 395+40),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


            textJobs = []
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                if w > dist and h > dist:
                    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    textJobs.append([img, f'{w/float(h):.2f}', (x,y), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0), 5])
                    textJobs.append([img, f'{w/float(h):.2f}', (x,y), cv.FONT_HERSHEY_PLAIN, 1, (0,0,255)])
            
            for job in textJobs:
                cv.putText(*job)

            if click is not None:
                cv.putText(img, f'({click[0]}, {click[1]})', (15, 90),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            cv.imshow('image', img)
        k = cv.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            break

    cv.destroyAllWindows()
    sys.exit("exiting")


doThing()
