import sys
import os
import cv2 as cv
import numpy as np
import pandas as pd

from trackbars import ChangeTrackingTrackbars
import ColorDescriptor
import Searcher

IMG_A = "./screenies/b_1440p.png"

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


def getRedMaskExpansion(imgHeight):
    # regression line fit for
    #
    # | Screen | Inventory  |
    # | height | tile width |
    # | ------ | ---------- |
    # |    768 |         66 |
    # |   1080 |         93 |
    # |   1440 |        126 |
    #
    return int(0.09 * imgHeight - 3)


def readTemplateFile(name):
    img = cv.imread(f'./images/{name}')
    return cv.resize(img, (60, 60), interpolation=cv.INTER_LANCZOS4)


class CompareImage(object):
    def __init__(self):
        self.minimum_commutative_image_diff = 1

    def compare_image(self, image_1, image_2):
        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            return commutative_image_diff
        return 10000  # random failure value

    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv.compareHist(
            first_image_hist, second_image_hist, cv.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv.matchTemplate(
            first_image_hist, second_image_hist, cv.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff


def doThing():
    img = cv.imread(IMG_A)

    templates = [f for f in os.listdir(
        'images') if os.path.isfile(os.path.join('images', f))]
    templateImages = [
        (os.path.splitext(template)[0], readTemplateFile(template)) for template in templates]

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
    maskThreshold = getRedMaskExpansion(img.shape[0])
    lower = np.clip([x - redTreshold for x in redSampleLab], 0, 255)
    upper = np.clip([x + redTreshold for x in redSampleLab], 0, 255)
    mask = cv.inRange(lab, lower, upper)

    mask = cv.dilate(mask, cv.getStructuringElement(
        cv.MORPH_RECT, (maskThreshold, maskThreshold)))

    (x, y, w, h) = cv.boundingRect(mask)
    mask[:] = (0,)
    mask[y:y+h, x:x+w] = (255,)

    tbs.changed = True

    tbs.add("rgb").min(0).max(2).initial(0).build()
    # tbs.add("d").min(1).max(50).initial(9).build()
    # tbs.add("sigmaColor").min(1).max(150).initial(75).build()
    # tbs.add("sigmaSpace").min(1).max(150).initial(75).build()
    tbs.add('alpha').min(0).max(300).initial(300).build()
    tbs.add('beta').min(-100).max(100).initial(100).build()
    tbs.addUint8('treshold').initial(50).build()
    tbs.add('kernel').min(1).max(49).initial(3).build()
    tbs.add('close').min(1).max(49).initial(5).build()

    imgLowRez = cv.resize(img, (1280, 720), interpolation=cv.INTER_LANCZOS4)
    maskLowRez = cv.resize(mask, (1280, 720), interpolation=cv.INTER_LINEAR)

    x, y, w, h = cv.boundingRect(maskLowRez)

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

            alpha = tbs.getPos('alpha') / 100.0
            beta = tbs.getPos('beta')

            cropped = imgLowRez[y:y+h, x:x+w]
            result = cropped

            # a = 2.52
            # b = -30
            result = cv.convertScaleAbs(result, alpha=alpha, beta=beta)

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

            #result = cv.GaussianBlur(result, (blur,blur), 0)
            result = cv.bilateralFilter(
                result, d=23, sigmaColor=83, sigmaSpace=89)

            result = cv.split(result)[tbs.getPos('rgb')]
            #result = (255-result)

            kernel = tbs.getPos('kernel')
            if kernel < 1:
                kernel = 1
            if kernel % 2 == 0:
                kernel += 1
            result = cv.Laplacian(result, cv.CV_64F, ksize=kernel)
            result = cv.convertScaleAbs(np.absolute(result))
            _, result = cv.threshold(result, tbs.getPos(
                'treshold'), 255, cv.THRESH_BINARY)

            close = tbs.getPos('close')
            if close % 2 == 0:
                close += 1

            result = cv.morphologyEx(result, cv.MORPH_CLOSE, cv.getStructuringElement(
                cv.MORPH_RECT, (close, close)))

            # Fill up gaps. Can't just take EXTERNAL contours because there's
            # rubbish because some corners aren't filled.
            contours, _ = cv.findContours(
                result, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                # fill all contour bounding rects as while
                cv.rectangle(result, (x, y), (x+w, y+h), 255, -1)

            # now get actual rectangle contours
            contours, _ = cv.findContours(
                result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            bboxes = [cv.boundingRect(cnt) for cnt in contours]

            def arc(rect):
                x, y, w, h = rect
                return w/float(h)
            # filter out cropped squares
            # (take only those whose aspect ratio is within 0.15 of our target "portrait" rectangle)
            bboxes = [b for b in bboxes if abs(0.85 - arc(b)) <= 0.15]

            result = cropped.copy()

            tiles = []
            i = 0
            for x, y, w, h in bboxes:
                tile = cropped[y:y+h, x:x+w]
                # grab middle bottom 60x60px square to extract
                # only the icon so that the scrolling text on the top
                # doesn't mess with our feature detection
                h, w, _ = tile.shape
                ystart = max(0, h - 61) if h > 60 else 0
                yend = ystart + 60
                xstart = (w - 60) // 2 if w > 60 else 0
                xend = xstart+60
                tile = tile[ystart:yend, xstart:xend]
                cv.imwrite(f'tiles/{i:02}.png', tile)
                tiles.append((tile, (x, y, w, h)))
                i += 1

            cd = ColorDescriptor.ColorDescriptor((8, 12, 13))

            featureIndex = []
            for templateName, template in templateImages:
                features = cd.describe(template)
                featureIndex.append((templateName, np.array(features)))

            s1 = Searcher.Searcher("")
            detected_tiles = {}
            for i, (tile, bbox) in enumerate(tiles):
                print(i)
                features = np.array(cd.describe(tile))
                results = s1.searchIdx(featureIndex, features)

                score, name = results[0]
                color = (0, 255, 0) if score < 0.7 else (0, 0, 255)
                cv.rectangle(
                    result, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)
                cv.putText(
                    result, f'{name}', (bbox[0], bbox[1]+15), cv.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0), 5)
                cv.putText(
                    result, f'{score:.2f}', (bbox[0], bbox[1]+15+15), cv.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 5)
                cv.putText(
                    result, f'{name}', (bbox[0], bbox[1]+15), cv.FONT_HERSHEY_PLAIN, 0.5, color, 1)
                cv.putText(
                    result, f'{score:.2f}', (bbox[0], bbox[1]+15+15), cv.FONT_HERSHEY_PLAIN, 0.7, color, 1)
                if score < 0.7:
                    detected_tiles[name] = detected_tiles.get(name, 0) + 1

            df = pd.read_csv('materials.csv', sep=',')
            materials = df.groupby('Name')
            
            print(detected_tiles)

            objs = []
            for name, amount in detected_tiles.items():
                group = materials.get_group(name)
                group = group.copy()
                group["Amount"] *= amount
                objs.append(group[['Material', 'Amount']])
            print(pd.concat(objs).groupby('Material').agg('sum'))

            # sift = cv.SIFT_create()
            # templates = []
            # for templateName, template in templateImages:
            #     kp, des = sift.detectAndCompute(template, None)
            #     templates.push((templateName, template, kp, des))

            # candidates = []
            # for tile in tiles:
            #     kp, des = sift.detectAndCompute(tile, None)
            #     candidates.push((tile, kp, des))

            # FLANN_INDEX_KDTREE = 1
            # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            # search_params = dict(checks=50)
            # flann = cv.FlannBasedMatcher(index_params, search_params)

            cv.imshow('image', result)
        k = cv.waitKey(1) & 0xFF
        if k == EXIT_KEY:
            break

    cv.destroyAllWindows()
    sys.exit("exiting")


doThing()
