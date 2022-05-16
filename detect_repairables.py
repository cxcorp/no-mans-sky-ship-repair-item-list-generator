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
    maskThreshold = 200
    lower = np.clip([x - redTreshold for x in redSampleLab], 0, 255)
    upper = np.clip([x + redTreshold for x in redSampleLab], 0, 255)
    mask = cv.inRange(lab, lower, upper)

    mask = cv.dilate(mask, cv.getStructuringElement(
        cv.MORPH_RECT, (maskThreshold, maskThreshold)))
    # img = cv.bitwise_and(img, img, mask=mask)

    #img = cv.medianBlur(img, ksize=3)

    click = None

    def mouseListener(event, x, y, flags, param):
        nonlocal click, tbs
        if event == cv.EVENT_LBUTTONDOWN:
            click = (x, y)
            tbs.changed = True

    cv.setMouseCallback('image', mouseListener)

    sift = cv.SIFT_create()

    img1 = cv.split(img)[2]
    img2 = cv.split(templateImages[0][0])[2]

    kp1, des1 = sift.detectAndCompute(img1, mask)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(0, 0, 255),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)

    while True:
        if tbs.changed:
            # print('changed')
            tbs.changed = False

            result = cv.drawMatchesKnn(
                img1, kp1, img2, kp2, matches, None, **draw_params)

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

            if click is not None:
                cv.putText(result, f'({click[0]}, {click[1]})', (15, 90),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            #cv.imshow('image', result)
            plt.imshow(result,)
            plt.show()
        k = cv.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            break

    cv.destroyAllWindows()
    sys.exit("exiting")


doThing()
