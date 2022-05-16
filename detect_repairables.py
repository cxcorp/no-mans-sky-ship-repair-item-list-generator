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
    # tbs.add("block").min(3).max(49).initial(3).build()
    # disti_max = 700
    # tbs.add("dist").min(0).max(disti_max).initial(10).build()

    #cv.imshow('image', img)

    blurred = cv.GaussianBlur(img, (3, 3), 0)
    lab = cv.cvtColor(blurred, cv.COLOR_BGR2Lab)
    redSample = (123, 0, 0)
    redSampleLab = rgb2lab(redSample)
    redTreshold = 4
    maskThreshold = 120
    lower = np.clip([x - redTreshold for x in redSampleLab], 0, 255)
    upper = np.clip([x + redTreshold for x in redSampleLab], 0, 255)
    mask = cv.inRange(lab, lower, upper)

    mask = cv.dilate(mask, cv.getStructuringElement(
        cv.MORPH_RECT, (maskThreshold, maskThreshold)))

    (x, y, w, h) = cv.boundingRect(mask)
    mask[:] = (0,)
    mask[y:y+h, x:x+w] = (255,)

    click = None

    def mouseListener(event, x, y, flags, param):
        nonlocal click, tbs
        if event == cv.EVENT_LBUTTONDOWN:
            click = (x, y)
            tbs.changed = True

    cv.setMouseCallback('image', mouseListener)

    tbs.changed = True

    tbs.add("rgb").min(0).max(2).initial(0).build()
    tbs.add("d").min(1).max(50).initial(9).build()
    tbs.add("sigmaColor").min(1).max(150).initial(75).build()
    tbs.add("sigmaSpace").min(1).max(150).initial(75).build()

    imgLowRez = cv.resize(img, (1280, 720), interpolation=cv.INTER_LANCZOS4)
    maskLowRez = cv.resize(mask, (1280, 720), interpolation=cv.INTER_LINEAR)

    i = 0
    cursors = ['|', '/', '-', '\\']

    def getCursor():
        nonlocal i
        idx = i % len(cursors)
        i = i+1
        return cursors[idx]
    while True:
        if tbs.changed:
            # print('changed')
            tbs.changed = False

            # offset = tbs.getPos("offset")
            # lower = np.clip([x - offset for x in graySample], 0, 255)
            # upper = np.clip([x + offset for x in graySample], 0, 255)
            # grayMask = cv.inRange(lab, lower, upper)
            #result = img.copy()

            #ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
            # cv.putText(img, f'dist%: {distRel:.3f}', (438, 395),
            #            cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            # cv.putText(img, f'dist: {dist:.3f}', (438, 395+40),
            #            cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            result = cv.bitwise_and(imgLowRez, imgLowRez, mask=maskLowRez)
            #channels = cv.split(result)
            # result = #channels[tbs.getPos("rgb")]

            d = tbs.getPos("d")
            sigmaColor = tbs.getPos("sigmaColor")
            sigmaSpace = tbs.getPos("sigmaSpace")
            #result = cv.GaussianBlur(result, (blur,blur), 0)
            result = cv.bilateralFilter(result, d, sigmaColor, sigmaSpace)

            cv.putText(result, getCursor(), (150, 150),
                       cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            # if click is not None:
            #     cv.putText(result, f'({click[0]}, {click[1]})', (15, 90),
            #                cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            cv.imshow('image', result)
        k = cv.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            break

    cv.destroyAllWindows()
    sys.exit("exiting")


doThing()
